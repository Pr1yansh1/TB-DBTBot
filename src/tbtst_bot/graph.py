# src/tbtst_bot/graph.py
from __future__ import annotations

import json
import logging
import random
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Set, Tuple, Optional

from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# RemoveMessage import location varies by langchain version
try:
    from langchain_core.messages import RemoveMessage  # type: ignore
except Exception:  # pragma: no cover
    from langchain.messages import RemoveMessage  # type: ignore

from .config import get_llm
from .prompts import load_prompt
from .rag_utils import retrieve_tb_docs


# =========================================================
# Logging / tracing knobs (env)
# =========================================================
# TBTST_DEBUG_METRICS=1
# TBTST_LOG_PROMPT_SIZES=1
# TBTST_LOG_LEVEL=INFO|DEBUG
LOG_LEVEL = os.getenv("TBTST_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("tbtst.graph")

DEBUG_METRICS = os.getenv("TBTST_DEBUG_METRICS", "0") == "1"
LOG_PROMPT_SIZES = os.getenv("TBTST_LOG_PROMPT_SIZES", "0") == "1"


# =========================================================
# Prompts (loaded from files)
# =========================================================
CLASSIFY_SYSTEM = load_prompt("classify_route.txt")
CLASSIFY_USER_TMPL = load_prompt("classify_route_user.txt")
CRISIS_TEXT = load_prompt("crisis.txt")

GLOBAL_RULES = load_prompt("global_response_rules.txt")

# DBT mini
DBT_SYSTEM = load_prompt("dbt_system.txt")

# DBT full
DBT_BRAIN_ROUTER_SYSTEM = load_prompt("dbt_brain.txt")
DBT_DT_SYSTEM = load_prompt("DBT/dt_prompt.txt")
DBT_MIND_SYSTEM = load_prompt("DBT/mind_prompt.txt")
DBT_ER_SYSTEM = load_prompt("DBT/er_prompt.txt")
DBT_IE_SYSTEM = load_prompt("DBT/ie_prompt.txt")

# TB RAG
TB_RAG_ANSWER_SYSTEM = load_prompt("tb_rag_answer_system.txt")
TB_RAG_ANSWER_USER_TMPL = load_prompt("tb_rag_answer_user.txt")

# Misc (meta / greetings / off-topic) - Spanish only
MISC_SYSTEM = """
Sos un asistente de apoyo para una app de TB y bienestar.
Tu tarea acá es responder mensajes "misc": saludos, dudas sobre el sistema, o cosas fuera de TB/DBT.
Respondé en español (vos), con calidez y brevedad.
Si el usuario pide algo concreto, pedí 1 pregunta breve para orientar. No uses jerga. No inventes datos médicos.
""".strip()


# =========================================================
# State
# =========================================================
RiskLevel = Literal["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]
Route = Literal["faq", "dbt", "misc"]

DBTModule = Literal["DT", "MIND", "ER", "IE"]
DBTMode = Literal["connect", "offer", "coach"]
DBTContinuity = Literal["same", "new", "unclear"]
DBTSkillStatus = Literal["none", "offered", "accepted", "declined", "paused", "completed"]


class State(MessagesState, total=False):
    # main routing
    safety_risk_level: RiskLevel
    safety_triggers: List[str]
    has_protective: bool
    safety_route: Literal["ok", "crisis"]
    route: Route
    route_source: str
    classifier_raw: str
    classifier_parse_ok: bool

    # dbt full router outputs
    dbt_module: DBTModule
    dbt_confidence: float
    dbt_rationale_brief: str
    dbt_signals: Dict[str, Any]
    dbt_router_raw: str

    # DBT conversational control (minimal but sufficient)
    dbt_mode: DBTMode
    dbt_continuity: DBTContinuity
    dbt_active_skill: str
    dbt_skill_status: DBTSkillStatus

    # short-term memory management (rolling summary)
    summary: str

    # optional: stored for debugging
    last_trace_id: str


# =========================================================
# Memory / context management settings
# =========================================================
MAX_TOKENS_BEFORE_SUMMARY = 4500
KEEP_LAST_MESSAGES = 8

LLM_RECENT_MESSAGES_MAX_TOKENS = 1800

SUMMARY_SYSTEM = """Sos un generador de resumen compacto y factual para un chatbot de apoyo (TB/DBT).
Objetivo: mantener continuidad y evitar preguntas repetidas.
Incluí SOLO:
- Hechos estables del usuario (si existen)
- Contexto clave del problema que compartió
- Qué se intentó/decidió
- Restricciones o detalles relevantes de seguridad (solo lo que el usuario dijo)
- Estado DBT (muy importante):
  - último módulo DBT activo (DT/MIND/ER/IE)
  - modo DBT actual (connect/offer/coach)
  - habilidad DBT activa (si hubo) y estado (offered/accepted/declined/paused/completed)
  - si el usuario pidió desahogarse sin habilidades
Sé breve. No inventes. Sin especulación.
Salida: SOLO el texto del resumen actualizado."""


# =========================================================
# Trace + metrics helpers
# =========================================================
def _trace_id_for_state(state: State) -> str:
    msgs: List[BaseMessage] = state.get("messages", []) or []
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            mid = getattr(m, "id", None)
            if mid:
                return str(mid)
            break
    return uuid.uuid4().hex[:10]


def _tokens(messages: List[BaseMessage]) -> int:
    return int(count_tokens_approximately(messages))


@dataclass
class CallMetrics:
    name: str
    trace_id: str
    latency_ms: float
    in_tokens: int
    out_chars: int


def _is_throttle_error(e: Exception) -> bool:
    msg = str(e)
    return ("ThrottlingException" in msg) or ("Too many requests" in msg) or ("Rate exceeded" in msg)


def _timed_invoke(name: str, llm: Any, messages: List[BaseMessage], *, trace_id: str) -> Tuple[Any, CallMetrics]:
    t0 = time.perf_counter()

    last_exc: Exception | None = None
    for attempt in range(6):  # 1 initial + 5 retries
        try:
            resp = llm.invoke(messages)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            m = CallMetrics(
                name=name,
                trace_id=trace_id,
                latency_ms=dt_ms,
                in_tokens=_tokens(messages),
                out_chars=len((getattr(resp, "content", "") or "")),
            )

            if DEBUG_METRICS:
                logger.info(
                    "[TRACE %s] [LLM] %s latency=%.1fms in_tokens~%d out_chars=%d",
                    m.trace_id,
                    m.name,
                    m.latency_ms,
                    m.in_tokens,
                    m.out_chars,
                )
            return resp, m

        except Exception as e:
            last_exc = e
            if _is_throttle_error(e) and attempt < 5:
                sleep_s = min(10.0, (2 ** attempt) * 0.6) + random.uniform(0.0, 0.35)
                logger.warning(
                    "[TRACE %s] [LLM] %s throttled; retrying in %.2fs (attempt %d/6)",
                    trace_id, name, sleep_s, attempt + 1
                )
                time.sleep(sleep_s)
                continue
            raise

    raise last_exc or RuntimeError("LLM invoke failed")


def _timed_tool(name: str, fn: Any, args: Dict[str, Any], *, trace_id: str) -> Any:
    t0 = time.perf_counter()
    out = fn.invoke(args)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if DEBUG_METRICS:
        n_sources = 0
        if isinstance(out, dict) and isinstance(out.get("sources"), list):
            n_sources = len(out["sources"])
        logger.info(
            "[TRACE %s] [TOOL] %s latency=%.1fms sources=%d",
            trace_id,
            name,
            dt_ms,
            n_sources,
        )
    return out


def _log_prompt_sizes_once() -> None:
    if not LOG_PROMPT_SIZES:
        return

    prompts: Dict[str, str] = {
        "GLOBAL_RULES": GLOBAL_RULES,
        "CLASSIFY_SYSTEM": CLASSIFY_SYSTEM,
        "DBT_SYSTEM": DBT_SYSTEM,
        "DBT_BRAIN_ROUTER_SYSTEM": DBT_BRAIN_ROUTER_SYSTEM,
        "DBT_DT_SYSTEM": DBT_DT_SYSTEM,
        "DBT_MIND_SYSTEM": DBT_MIND_SYSTEM,
        "DBT_ER_SYSTEM": DBT_ER_SYSTEM,
        "DBT_IE_SYSTEM": DBT_IE_SYSTEM,
        "TB_RAG_ANSWER_SYSTEM": TB_RAG_ANSWER_SYSTEM,
    }
    for name, text in prompts.items():
        lines = len((text or "").splitlines())
        chars = len(text or "")
        toks = _tokens([SystemMessage(content=text or "")])
        logger.info("[PROMPT_FILE] %s lines=%d chars=%d tokens~%d", name, lines, chars, toks)


# =========================================================
# Core helpers
# =========================================================
def _latest_user_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _safe_fill_user_text(template: str, user_text: str) -> str:
    return template.replace("{user_text}", user_text)


class ClassifyOut(BaseModel):
    safety_risk_level: RiskLevel = Field(...)
    safety_triggers: List[str] = Field(default_factory=list)
    has_protective: bool = Field(False)
    route: Route = Field(...)


def _coerce_and_validate_classifier(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    try:
        obj = ClassifyOut.model_validate(data)
        triggers = [t for t in (obj.safety_triggers or []) if isinstance(t, str)]
        out = {
            "safety_risk_level": obj.safety_risk_level,
            "safety_triggers": triggers[:5],
            "has_protective": bool(obj.has_protective),
            "route": obj.route,
        }
        return out, True
    except ValidationError:
        return {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "misc",
        }, False


def _parse_classifier_json(raw: str) -> Tuple[Dict[str, Any], bool]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return _coerce_and_validate_classifier({})
        return _coerce_and_validate_classifier(data)
    except Exception:
        return _coerce_and_validate_classifier({})


def _format_sources_block(sources: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for s in sources:
        sid = s.get("id")
        title = s.get("title", "Unknown")
        filename = s.get("filename", "unknown.txt")
        excerpt = (s.get("excerpt") or "").strip()
        parts.append(
            f"Fuente {sid}:\n"
            f"Título: {title}\n"
            f"Archivo: {filename}\n"
            f"Extracto:\n{excerpt}\n"
        )
    return "\n".join(parts).strip()


def _render_answer_with_references(answer_text: str, sources: List[Dict[str, Any]], citations_used: List[int]) -> str:
    src_by_id: Dict[int, Dict[str, Any]] = {}
    for s in sources:
        try:
            sid = int(s.get("id"))
            src_by_id[sid] = s
        except Exception:
            continue

    used: Set[int] = set()
    for x in citations_used or []:
        if isinstance(x, int) and x in src_by_id:
            used.add(x)

    if not used:
        return (answer_text or "").strip()

    lines = [(answer_text or "").strip(), "", "Referencias:"]
    for sid in sorted(used):
        s = src_by_id[sid]
        title = s.get("title", "Unknown")
        filename = s.get("filename", "unknown.txt")
        lines.append(f"[{sid}] {title} ({filename})")
    return "\n".join(lines).strip()


def _trim_for_llm(messages: List[BaseMessage], *, max_tokens: int) -> List[BaseMessage]:
    if not messages:
        return []
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "tool"),
    )


def _system_and_summary_messages(system_prompt: str, summary: str) -> List[BaseMessage]:
    combined_system = f"{GLOBAL_RULES.strip()}\n\n---\n\n{system_prompt.strip()}".strip()
    msgs: List[BaseMessage] = [SystemMessage(content=combined_system)]
    if summary.strip():
        msgs.append(
            SystemMessage(
                content=(
                    "Memoria de conversación (concisa y factual). Usala para mantener continuidad:\n"
                    f"{summary.strip()}"
                )
            )
        )
    return msgs


def _llm_messages_with_summary(system_prompt: str, state: State, *, trace_id: str, label: str) -> List[BaseMessage]:
    summary = (state.get("summary") or "").strip()
    prefix = _system_and_summary_messages(system_prompt, summary)

    base_tokens = _tokens(prefix)
    recent_budget = max(256, LLM_RECENT_MESSAGES_MAX_TOKENS - base_tokens)
    recent = _trim_for_llm(state.get("messages", []), max_tokens=recent_budget)

    if DEBUG_METRICS:
        sys_tokens = _tokens([prefix[0]])
        summ_tokens = _tokens(prefix[1:]) if len(prefix) > 1 else 0
        recent_tokens = _tokens(recent)
        total = sys_tokens + summ_tokens + recent_tokens
        logger.info(
            "[TRACE %s] [PROMPT:%s] sys~%d summary~%d recent~%d recent_budget~%d total~%d",
            trace_id,
            label,
            sys_tokens,
            summ_tokens,
            recent_tokens,
            int(recent_budget),
            total,
        )

    return [*prefix, *recent]


def _dbt_state_system_note(state: State) -> SystemMessage:
    """
    Nota de estado interno DBT (NO visible al usuario). Esto reemplaza regex:
    el router y el agente usan esta información para decidir modo/continuidad.
    """
    mode: str = state.get("dbt_mode", "connect")
    cont: str = state.get("dbt_continuity", "new")
    active_skill: str = (state.get("dbt_active_skill") or "").strip()
    skill_status: str = state.get("dbt_skill_status", "none")
    return SystemMessage(
        content=(
            "Estado interno DBT (para continuidad):\n"
            f"- modo_actual: {mode}\n"
            f"- continuidad_previa: {cont}\n"
            f"- habilidad_activa: {active_skill or 'ninguna'}\n"
            f"- estado_habilidad: {skill_status}\n"
            "Reglas duras:\n"
            "- Si estado_habilidad=declined o paused: NO ofrezcas nuevas habilidades; usá modo=connect.\n"
            "- Si estado_habilidad=accepted y la respuesta del usuario es corta: usá modo=coach y continuidad=same.\n"
        )
    )


def _dbt_mode_instructions(state: State) -> SystemMessage:
    """
    Instrucciones cortas y fuertes para impedir info-dumps.
    """
    mode: DBTMode = state.get("dbt_mode", "connect")
    active_skill = (state.get("dbt_active_skill") or "").strip()

    if mode == "connect":
        rules = (
            "MODO=CONNECT (conversación, sin info-dump):\n"
            "- Validá/empathizá en 1–2 frases.\n"
            "- Hacé EXACTAMENTE 1 pregunta breve.\n"
            "- NO enseñes habilidades DBT, NO listes pasos, NO ofrezcas múltiples opciones.\n"
            "- Máximo ~120–180 palabras.\n"
        )
    elif mode == "offer":
        rules = (
            "MODO=OFFER (ofrecer 1 habilidad, corto):\n"
            "- Ofrecé UNA sola habilidad DBT (nombre + definición 1 línea).\n"
            "- Da 2–3 pasos máximos en viñetas.\n"
            "- Cerrá con 1 pregunta de consentimiento (si quiere probar ahora).\n"
            "- No agregues una segunda habilidad.\n"
            "- Máximo ~180–250 palabras.\n"
        )
    else:  # coach
        rules = (
            "MODO=COACH (continuar la misma habilidad):\n"
            f"- Continuá SOLO con la habilidad ya activa: {active_skill or '(si no hay, elegí 1 y mantenela)'}.\n"
            "- Da UN solo siguiente paso o mini-ejercicio (2–4 viñetas).\n"
            "- Cerrá con 1 check-in.\n"
            "- Máximo ~220–320 palabras.\n"
        )

    return SystemMessage(content=rules)


def _dbt_max_tokens_for_mode(state: State) -> int:
    mode: DBTMode = state.get("dbt_mode", "connect")
    if mode == "connect":
        return 260
    if mode == "offer":
        return 420
    return 520


# =========================================================
# Nodes: classify + safety
# =========================================================
def classify_node(state: State) -> Dict[str, Any]:
    trace_id = _trace_id_for_state(state)
    state["last_trace_id"] = trace_id

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        if DEBUG_METRICS:
            logger.info("[TRACE %s] classify empty input -> route=misc", trace_id)
        return {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "safety_route": "ok",
            "route": "misc",
            "route_source": "empty_input_default",
            "classifier_raw": "",
            "classifier_parse_ok": True,
        }

    user_prompt = _safe_fill_user_text(CLASSIFY_USER_TMPL, user_text)
    llm = get_llm(temperature=0.0, max_tokens=220)

    raw_msg, _ = _timed_invoke(
        "classify:first",
        llm,
        [SystemMessage(content=CLASSIFY_SYSTEM), HumanMessage(content=user_prompt)],
        trace_id=trace_id,
    )
    raw1 = (raw_msg.content or "").strip()
    parsed1, ok1 = _parse_classifier_json(raw1)

    need_retry = (not ok1) or (parsed1.get("safety_risk_level") == "uncertain" and not (parsed1.get("safety_triggers") or []))

    raw = raw1
    parsed = parsed1
    ok = ok1

    if need_retry:
        retry_system = CLASSIFY_SYSTEM.strip() + "\n\nREMINDER: Output EXACTLY one line of STRICT JSON with only the required keys. No extra text."
        raw_msg2, _ = _timed_invoke(
            "classify:retry",
            llm,
            [SystemMessage(content=retry_system), HumanMessage(content=user_prompt)],
            trace_id=trace_id,
        )
        raw2 = (raw_msg2.content or "").strip()
        parsed2, ok2 = _parse_classifier_json(raw2)

        if ok2:
            raw, parsed, ok = raw2, parsed2, ok2
        else:
            raw, parsed, ok = raw1, {"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "misc"}, False

    risk: RiskLevel = parsed["safety_risk_level"]
    route: Route = parsed["route"]

    safety_route: Literal["ok", "crisis"] = "crisis" if risk in ("passive", "active_no_plan", "active_with_plan") else "ok"

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] classify result risk=%s safety_route=%s route=%s parse_ok=%s triggers=%s",
            trace_id, risk, safety_route, route, ok, parsed.get("safety_triggers", []),
        )

    return {
        "safety_risk_level": risk,
        "safety_triggers": parsed["safety_triggers"],
        "has_protective": parsed["has_protective"],
        "safety_route": safety_route,
        "route": route,
        "route_source": "llm",
        "classifier_raw": raw,
        "classifier_parse_ok": ok,
    }


def crisis_node(state: State) -> Dict[str, Any]:
    return {"messages": [AIMessage(content=CRISIS_TEXT)]}


# =========================================================
# Memory manager node (summarize + delete)
# =========================================================
def memory_manager_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    messages: List[BaseMessage] = state.get("messages", [])
    if not messages:
        return {}

    approx_tokens = int(count_tokens_approximately(messages))
    if DEBUG_METRICS:
        logger.info("[TRACE %s] memory check messages=%d tokens~%d", trace_id, len(messages), approx_tokens)

    if approx_tokens < MAX_TOKENS_BEFORE_SUMMARY:
        return {}

    existing_summary = (state.get("summary") or "").strip()
    if existing_summary:
        summary_message = (
            f"Resumen existente:\n{existing_summary}\n\n"
            "Actualizalo incorporando SOLO la información nueva de los mensajes recientes:"
        )
    else:
        summary_message = "Creá un resumen conciso de la conversación anterior:"

    summarizer = get_llm(temperature=0.0, max_tokens=320)
    summary_input = [SystemMessage(content=SUMMARY_SYSTEM), *messages, HumanMessage(content=summary_message)]

    resp, _ = _timed_invoke("summarize", summarizer, summary_input, trace_id=trace_id)
    new_summary = (resp.content or "").strip()

    to_remove: List[Any] = []
    if len(messages) > KEEP_LAST_MESSAGES:
        for msg in messages[:-KEEP_LAST_MESSAGES]:
            mid = getattr(msg, "id", None)
            if mid:
                to_remove.append(RemoveMessage(id=mid))

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] memory summarized -> new_summary_chars=%d removed=%d (kept=%d)",
            trace_id, len(new_summary), len(to_remove), KEEP_LAST_MESSAGES,
        )

    out: Dict[str, Any] = {"summary": new_summary}
    if to_remove:
        out["messages"] = to_remove
    return out


# =========================================================
# DBT MINI node
# =========================================================
def dbt_mini_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    llm = get_llm(temperature=0.25, max_tokens=600)
    msgs = _llm_messages_with_summary(DBT_SYSTEM, state, trace_id=trace_id, label="dbt_mini")
    reply, _ = _timed_invoke("dbt:mini", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# TB-info specialist: citation-safe RAG
# =========================================================
class TBAnswerJSON(BaseModel):
    answer: str = Field(...)
    citations_used: List[int] = Field(default_factory=list)


def tb_info_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {"messages": [AIMessage(content="")]}

    tool_out = _timed_tool("retrieve_tb_docs", retrieve_tb_docs, {"query": user_text}, trace_id=trace_id)
    sources = tool_out.get("sources", []) if isinstance(tool_out, dict) else []

    sources_block = _format_sources_block(sources)
    user_prompt = TB_RAG_ANSWER_USER_TMPL.replace("{user_text}", user_text).replace("{sources_block}", sources_block)

    summary = (state.get("summary") or "").strip()
    prefix = _system_and_summary_messages(TB_RAG_ANSWER_SYSTEM, summary)

    llm = get_llm(temperature=0.2, max_tokens=700).with_structured_output(TBAnswerJSON)

    t0 = time.perf_counter()
    result = llm.invoke([*prefix, HumanMessage(content=user_prompt)])
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] [LLM] tb_info latency=%.1fms in_tokens~%d out_answer_chars=%d cited=%s",
            trace_id,
            dt_ms,
            _tokens([*prefix, HumanMessage(content=user_prompt)]),
            len((result.answer or "")),
            result.citations_used,
        )

    final_text = _render_answer_with_references(answer_text=result.answer, sources=sources, citations_used=result.citations_used)
    return {"messages": [AIMessage(content=final_text)]}


# =========================================================
# MISC node
# =========================================================
def misc_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    llm = get_llm(temperature=0.2, max_tokens=350)
    msgs = _llm_messages_with_summary(MISC_SYSTEM, state, trace_id=trace_id, label="misc")
    reply, _ = _timed_invoke("misc", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# DBT FULL subgraph (router -> module agent)
# =========================================================
class DBTBrainSignals(BaseModel):
    emotion_intensity: Literal["low", "medium", "high"]
    impulse_or_urge_present: bool
    problem_solvable_now: bool
    interpersonal_context: bool
    attention_or_judgment_issue: bool


class DBTBrainOut(BaseModel):
    # Module decision
    module: DBTModule
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_brief: str
    signals: DBTBrainSignals

    # Conversational control decision (NO user-facing content)
    mode: DBTMode
    continuity: DBTContinuity

    # Minimal thread memory updates (router decides; no regex)
    active_skill: str = Field(default="")
    skill_status: DBTSkillStatus = Field(default="none")


def dbt_brain_router_node(state: State) -> Dict[str, Any]:
    """
    DBT router is now context-aware and also controls mode/continuity.
    It sees summary + recent messages + internal DBT state note.
    """
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {
            "dbt_module": "MIND",
            "dbt_confidence": 0.0,
            "dbt_rationale_brief": "Entrada vacía.",
            "dbt_signals": {},
            "dbt_router_raw": "",
            "dbt_mode": "connect",
            "dbt_continuity": "unclear",
            "dbt_active_skill": state.get("dbt_active_skill", ""),
            "dbt_skill_status": state.get("dbt_skill_status", "none"),
        }

    llm = get_llm(temperature=0.0, max_tokens=280).with_structured_output(DBTBrainOut)

    # Build messages like other agents: system+summary+recent, but add DBT state note.
    base_msgs = _llm_messages_with_summary(DBT_BRAIN_ROUTER_SYSTEM, state, trace_id=trace_id, label="dbt_brain")
    msgs = [base_msgs[0], _dbt_state_system_note(state), *base_msgs[1:]]  # keep summary+recent after note

    t0 = time.perf_counter()
    result: DBTBrainOut = llm.invoke(msgs)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] [LLM] dbt_brain latency=%.1fms in_tokens~%d module=%s mode=%s cont=%s status=%s skill=%s",
            trace_id,
            dt_ms,
            _tokens(msgs),
            result.module,
            result.mode,
            result.continuity,
            result.skill_status,
            (result.active_skill or ""),
        )

    # Hard override: if state says declined/paused, force connect to avoid pushing skills
    prev_status: DBTSkillStatus = state.get("dbt_skill_status", "none")
    forced_mode: Optional[DBTMode] = None
    if prev_status in ("declined", "paused"):
        forced_mode = "connect"

    final_mode: DBTMode = forced_mode or result.mode
    final_status: DBTSkillStatus = result.skill_status
    final_skill: str = (result.active_skill or "").strip()

    return {
        "dbt_module": result.module,
        "dbt_confidence": float(result.confidence),
        "dbt_rationale_brief": result.rationale_brief,
        "dbt_signals": result.signals.model_dump(),
        "dbt_router_raw": result.model_dump_json(),
        "dbt_mode": final_mode,
        "dbt_continuity": result.continuity,
        "dbt_active_skill": final_skill or state.get("dbt_active_skill", ""),
        "dbt_skill_status": final_status,
    }


def _dbt_module_agent(system_prompt: str, state: State, *, label: str) -> Dict[str, Any]:
    """
    Module agent gets:
    - GLOBAL_RULES + module prompt
    - summary
    - DBT internal state note
    - mode instructions (connect/offer/coach)
    This prevents info dumping without extra nodes.
    """
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    max_tokens = _dbt_max_tokens_for_mode(state)
    llm = get_llm(temperature=0.25, max_tokens=max_tokens)

    base_msgs = _llm_messages_with_summary(system_prompt, state, trace_id=trace_id, label=label)

    # Inject mode instructions + DBT state note right after the main system prompt
    msgs = [
        base_msgs[0],
        _dbt_mode_instructions(state),
        _dbt_state_system_note(state),
        *base_msgs[1:],
    ]

    reply, _ = _timed_invoke(f"dbt:{label}", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


def dbt_dt_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_DT_SYSTEM, state, label="DT")


def dbt_mind_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_MIND_SYSTEM, state, label="MIND")


def dbt_er_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_ER_SYSTEM, state, label="ER")


def dbt_ie_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_IE_SYSTEM, state, label="IE")


def branch_after_dbt_brain(state: State) -> DBTModule:
    return state.get("dbt_module", "MIND")


def build_dbt_full_subgraph():
    sg = StateGraph(State)
    sg.add_node("dbt_brain", dbt_brain_router_node)
    sg.add_node("DT", dbt_dt_node)
    sg.add_node("MIND", dbt_mind_node)
    sg.add_node("ER", dbt_er_node)
    sg.add_node("IE", dbt_ie_node)

    sg.add_edge(START, "dbt_brain")
    sg.add_conditional_edges("dbt_brain", branch_after_dbt_brain, {"DT": "DT", "MIND": "MIND", "ER": "ER", "IE": "IE"})
    sg.add_edge("DT", END)
    sg.add_edge("MIND", END)
    sg.add_edge("ER", END)
    sg.add_edge("IE", END)

    return sg.compile()


DBT_FULL_SUBGRAPH = build_dbt_full_subgraph()


# =========================================================
# Top-level routing
# =========================================================
def branch_after_classify(state: State) -> Literal["crisis", "tb_info", "dbt", "misc"]:
    if state.get("safety_route") == "crisis":
        return "crisis"

    route = state.get("route", "misc")
    if route == "faq":
        return "tb_info"
    if route == "dbt":
        return "dbt"
    return "misc"


def build_graph(*, dbt_node: Any):
    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("memory", memory_manager_node)

    g.add_node("crisis", crisis_node)
    g.add_node("tb_info", tb_info_node)
    g.add_node("dbt", dbt_node)
    g.add_node("misc", misc_node)

    g.add_edge(START, "classify")
    g.add_edge("classify", "memory")

    g.add_conditional_edges(
        "memory",
        branch_after_classify,
        {"crisis": "crisis", "tb_info": "tb_info", "dbt": "dbt", "misc": "misc"},
    )

    g.add_edge("crisis", END)
    g.add_edge("tb_info", END)
    g.add_edge("dbt", END)
    g.add_edge("misc", END)

    return g.compile(checkpointer=MemorySaver())


# Public compiled graphs
GRAPH_DBT_MINI = build_graph(dbt_node=dbt_mini_node)
GRAPH_DBT_FULL = build_graph(dbt_node=DBT_FULL_SUBGRAPH)


_log_prompt_sizes_once()

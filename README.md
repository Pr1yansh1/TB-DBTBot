### `system_prompts.json`

**What it contains**

All the global and agent-level prompt templates for the system, including:

* Safety classifier:

  * `safety_classifier_system`
  * `safety_classifier_user_template`

* Domain router:

  * `domain_router_system`
  * `domain_router_user_template`

* FAQ agent:

  * `faq_system`
  * `faq_with_kb_user_template`
  * `faq_no_kb_user_template`

* DBT mode router:

  * `dbt_mode_router_system`
  * `dbt_mode_router_user_template`

* DBT micro-coach prompts:

  * `dbt_modes` → `mindfulness`, `distress`, `emotion`, `interpersonal`, each with `"system_prompt"` + `"user_template"`

* Psychoeducation agent:

  * `psychoed_system`
  * `psychoed_user_template`

* Crisis handoff template:

  * `crisis_message_template`

* Heuristic keyword buckets for DBT routing:

  * `dbt_distress_keywords`, `dbt_interpersonal_keywords`, `dbt_emotion_keywords`, `dbt_mindfulness_keywords`

**How it’s used**

* Loaded once via `load_system_prompts()` in:

  * `safety.py` (safety LLM classifier)
  * `routing.py` (domain router LLM)
  * `agents/faq_agent.py` (FAQ system + templates)
  * `agents/dbt_agent.py` (DBT mode router + mode-specific prompts)
  * `agents/psycho_agent.py` (psychoeducation prompts)
  * `graph.py` (crisis message template)

Acts as the single source of truth for tone, style, task instructions, and JSON-output formats for each LLM call.

---

### `tb_faq_kb.json`

**What it contains**

A mini knowledge base of TB FAQ entries, each typically like:

```jsonc
{
  "id": "faq_treatment_length",
  "question": "How long do I need to take TB treatment?",
  "answer": "Clinic-approved explanation...",
  "keywords": ["how long", "treatment length", "months", "duration"]
}
```

**How it’s used**

* Loaded via `load_tb_faq_kb()` in `agents/faq_agent.py`.
* `_retrieve_best_faq` (or a future `_retrieve_top_k_faq`) does lexical matching over `keywords` + question text.
* If a match is found, the FAQ agent:

  * Uses `faq_with_kb_user_template` to adapt the canonical `answer` to the patient’s message.
* If no match is found, it falls back to a more general FAQ response with `faq_no_kb_user_template`.

---

### `dbt_skills_kb.json`

**What it contains**

A mini DBT skills knowledge base, structured by mode:

```jsonc
{
  "mindfulness":   [ { "id": "...", "name": "...", "summary": "...", "steps": [/* ... */] } ],
  "distress":      [ /* ... */ ],
  "emotion":       [ /* ... */ ],
  "interpersonal": [ /* ... */ ]
}
```

(Names, summaries, and steps are distilled from DBT overview material.)

**How it’s used**

* Loaded via `load_dbt_skills_kb()` in `agents/dbt_agent.py`.
* After selecting a DBT `mode` (via heuristic + LLM), the agent:

  * Pulls the relevant skills for that mode.
  * Injects them as JSON (`skills_json=...`) into the mode-specific `user_template`.
* The LLM then picks 1–2 appropriate skills and turns them into a tailored, step-by-step response.

---

### `safety_keywords.json`

**What it contains**

A set of lexical patterns for safety triage, e.g.:

* Phrases for suicidal ideation, self-harm, hopelessness, etc. in English and Spanish.
* Optional “protective-factor” phrases (family, faith, fear of death, reasons to live).

**How it’s used**

* Loaded in `safety.py` via a helper like `load_safety_keywords()`.
* Used for:

  * Heuristic safety classification (rule-based layer) before/alongside the LLM safety classifier.
  * Deciding whether to escalate to the crisis path vs. proceed to the router.

The idea: obvious high-risk phrases trigger crisis even if the LLM misclassifies or is uncertain.

---

### `routing_keywords.json`

**What it contains**

Keyword lists used to steer domain routing and/or DBT mode selection, e.g.:

* FAQ-ish language: “dose”, “appointment”, “lab”, “results”, “side effects”.
* DBT-ish language: “panic”, “anxious”, “fight with my husband”, “stressed”, etc.
* Psychoeducation-ish language: “what is TB”, “how does it spread”, “why so many pills”.

**How it’s used**

* Loaded in `routing.py` via `load_routing_keywords()`.
* Used in simple heuristics:

  * If message contains meds/logistics words → route to `faq`.
  * If it’s emotion/relationship heavy → route to `dbt`.
  * If it’s explanation-focused → route to `psychoed`.

These heuristics run before (or alongside) the LLM router, to reduce unnecessary LLM calls and make behavior more predictable.

---

### Quick mental model

* `system_prompts.json` – how agents and routers think and speak.
* `tb_faq_kb.json` – canonical TB facts and clinic-aligned answers.
* `dbt_skills_kb.json` – small DBT skills library for micro-coaching.
* `safety_keywords.json` – hard-coded red flags for risk detection.
* `routing_keywords.json` – keywords to push messages to FAQ / DBT / psycho-ed.


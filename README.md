Explanation of Keyword Files

keywords/safety_crisis.txt
This file contains explicit, high-risk phrases associated with suicide, self-harm, or harm to others. During routing, the safety gate scans user messages for any of these terms. If a match is found, the system immediately routes the conversation to the Crisis Bot and bypasses all other logic. These keywords act as hard triggers for elevated-risk responses.

keywords/safety_intensifiers.txt
These intensifiers amplify the severity of an otherwise ambiguous or weak risk signal (e.g., “I’m done,” “right now,” “can’t take it anymore”). The safety classifier checks for combinations of crisis terms + intensifiers to detect medium-risk cases where the user is not explicitly suicidal but shows escalating emotional distress. These keywords help reduce missed-risk cases.

keywords/domain_faq.txt
This file holds TB-specific vocabulary—treatment logistics, symptoms, side-effects, lab terms, clinic workflow terms, etc. If these appear in the user message (and safety triggers are absent), the router directs the message to the FAQ agent. This ensures practical medical/treatment questions do not get misinterpreted as emotional concerns.

keywords/domain_dbt.txt
These are emotional, interpersonal, and urge-related keywords. When the router finds these expressions (again, in the absence of a safety trigger), it sends the message to the DBT coaching pipeline. This file anchors the system’s ability to distinguish coping/support needs from clinical TB questions.

import os
import requests
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_synthetic_telemetry(issue_body: str, pr_body: str) -> dict:
    """
    Takes the human-written Issue and PR, and forces the LLM to reverse-engineer
    it into a raw stack trace and a structured RCA JSON.
    """
    if not GROQ_API_KEY:
         raise ValueError("GROQ_API_KEY is missing.")

    # THE 0.1% PROMPT: Strict roleplay, precise constraints, explicit schema mapping.
    system_prompt = """You are an elite AI Data Engineer constructing a dataset for a fine-tuning pipeline. 
Your objective is to ingest human-written bug reports and Pull Request resolutions, and reverse-engineer them into two distinct outputs:
1. 'raw_log': A highly realistic, messy, unstructured Java/Elasticsearch crash log, stack trace, or stderr dump that represents the EXACT symptom of the bug. Include realistic timestamps, thread IDs, and exception paths.
2. 'rca_schema': A strict JSON object representing the Root Cause Analysis.

You MUST output ONLY a valid JSON object with exactly these two keys.
The 'rca_schema' MUST contain exactly these keys: severity, failing_service, error_code, root_cause_analysis, recommended_remediation.
"""

    user_prompt = f"""
ORIGINAL ISSUE REPORT:
{issue_body}

PULL REQUEST RESOLUTION:
{pr_body}

Generate the JSON object containing the 'raw_log' and the 'rca_schema'.
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # THIS forces the syntax constraint
        "response_format": {"type": "json_object"},
        "temperature": 0.2, # Low temperature for deterministic, factual extraction
        "max_tokens": 2048
    }
    
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status() # Automatically raises an error for 4xx/5xx responses
        
        content = resp.json()["choices"][0]["message"]["content"]
        extracted_data = json.loads(content)
        return extracted_data
    except Exception as e:
        logger.exception(f"LLM Transformation Failed: {e}")
        return None
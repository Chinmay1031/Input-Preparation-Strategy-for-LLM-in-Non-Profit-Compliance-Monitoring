"""
caller.py
---------
Single LLM caller used by all four strategies.

The prompt is fixed. The model is fixed. The temperature is fixed.
The only thing that changes per call is the prepared document text.
This controlled design is what makes the experiment valid.

All results are saved to disk after one run — the API is never
called again during analysis or evaluation.
"""

import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.llm.prompt_template import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Experiment constants ──────────────────────────────────────────────────────
MODEL       = "gpt-4o"
TEMPERATURE = 0.3    # low but non-zero — captures natural variance for
                     # consistency measurement without being deterministic
MAX_TOKENS  = 1000   # sufficient for structured JSON verdict
N_RUNS      = 3      # repeated runs per strategy per document


def call_llm(prepared_text: str) -> dict:
    """
    Single LLM call with structured JSON output enforced.

    Args:
        prepared_text: the document content prepared by one of the
                       four strategies (S1, S2, S3, S4)

    Returns:
        dict containing:
            verdict:       parsed JSON compliance verdict
            tokens_input:  number of input tokens used
            tokens_output: number of output tokens used
            tokens_total:  total tokens
            model:         model name from API response
            raw_response:  raw text from API (for debugging)
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},  # enforces JSON output
        messages=[
            {
                "role":    "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role":    "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    document_content=prepared_text
                )
            }
        ]
    )

    raw_text = response.choices[0].message.content

    # Parse JSON verdict
    try:
        verdict = json.loads(raw_text)
    except json.JSONDecodeError:
        # If JSON parsing fails, store the error and raw text
        verdict = {
            "parse_error":   True,
            "raw_response":  raw_text,
            "revenue_concentration": "CLEAR",
            "expense_spike":         "CLEAR",
            "passthrough_risk":      "CLEAR",
            "unallowable_expenditure": "CLEAR",
            "audit_opinion":         "CLEAR",
            "going_concern":         "CLEAR",
            "overall_verdict":       "UNKNOWN",
            "flags":                 [],
            "confidence":            0.0
        }

    return {
        "verdict":        verdict,
        "tokens_input":   response.usage.prompt_tokens,
        "tokens_output":  response.usage.completion_tokens,
        "tokens_total":   response.usage.total_tokens,
        "model":          response.model,
        "raw_response":   raw_text,
    }


def call_llm_with_retry(prepared_text: str, max_retries: int = 3) -> dict:
    """
    Wrapper with exponential backoff retry for rate limit errors.
    """
    for attempt in range(max_retries):
        try:
            return call_llm(prepared_text)
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait = 2 ** attempt
                print(f"    Rate limit hit — waiting {wait}s before retry {attempt + 1}")
                time.sleep(wait)
            elif attempt == max_retries - 1:
                # Final attempt failed — return error result
                return {
                    "verdict": {
                        "parse_error":             True,
                        "error_message":           error_msg,
                        "revenue_concentration":   "CLEAR",
                        "expense_spike":           "CLEAR",
                        "passthrough_risk":        "CLEAR",
                        "unallowable_expenditure": "CLEAR",
                        "audit_opinion":           "CLEAR",
                        "going_concern":           "CLEAR",
                        "overall_verdict":         "ERROR",
                        "flags":                   [],
                        "confidence":              0.0
                    },
                    "tokens_input":  0,
                    "tokens_output": 0,
                    "tokens_total":  0,
                    "model":         MODEL,
                    "raw_response":  error_msg,
                }
            else:
                time.sleep(1)

    return call_llm(prepared_text)
"""
caller.py
---------
Single LLM caller used by all four strategies.

The prompt is fixed. The model is fixed. The temperature is fixed.
The only thing that changes per call is the prepared document text.
This controlled design is what makes the experiment valid.

Documents that exceed the account's per-request token limit are
truncated rather than dropped, and the truncation is recorded. The
fact that full-document prompting cannot fit large grantee documents
within standard API rate limits is itself a finding relevant to the
practical feasibility of that approach.
"""

import os
import json
import time
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from src.llm.prompt_template import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Experiment constants ──────────────────────────────────────────────────────
MODEL       = "gpt-4o"
TEMPERATURE = 0.3
MAX_TOKENS  = 1000
N_RUNS      = 3

# Must stay below the account's tokens-per-minute limit (30,000)
MAX_INPUT_TOKENS = 22_000
INTER_CALL_DELAY = 3

_ENCODING = tiktoken.encoding_for_model("gpt-4o")


def _truncate_to_limit(text: str, max_tokens: int = MAX_INPUT_TOKENS):
    """
    Truncate text to fit within the per-request token limit.
    Returns (text, was_truncated, original_token_count).
    """
    tokens   = _ENCODING.encode(text)
    original = len(tokens)

    if original <= max_tokens:
        return text, False, original

    keep      = max_tokens - 1_500
    truncated = _ENCODING.decode(tokens[:keep])
    truncated += (
        "\n\n[DOCUMENT TRUNCATED: the source document exceeded the "
        "maximum request size and was cut off at this point.]"
    )
    return truncated, True, original


def call_llm(prepared_text: str) -> dict:
    """Single LLM call with structured JSON output enforced."""
    prepared_text, was_truncated, original_tokens = _truncate_to_limit(
        prepared_text
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                document_content=prepared_text
            )}
        ]
    )

    raw_text = response.choices[0].message.content

    try:
        verdict = json.loads(raw_text)
    except json.JSONDecodeError:
        verdict = {
            "parse_error":             True,
            "raw_response":            raw_text,
            "revenue_concentration":   "CLEAR",
            "expense_spike":           "CLEAR",
            "passthrough_risk":        "CLEAR",
            "unallowable_expenditure": "CLEAR",
            "audit_opinion":           "CLEAR",
            "going_concern":           "CLEAR",
            "overall_verdict":         "UNKNOWN",
            "flags":                   [],
            "confidence":              0.0
        }

    return {
        "verdict":         verdict,
        "tokens_input":    response.usage.prompt_tokens,
        "tokens_output":   response.usage.completion_tokens,
        "tokens_total":    response.usage.total_tokens,
        "model":           response.model,
        "raw_response":    raw_text,
        "was_truncated":   was_truncated,
        "original_tokens": original_tokens,
    }


def call_llm_with_retry(prepared_text: str, max_retries: int = 4) -> dict:
    """
    Wrapper with exponential backoff for transient rate limits.
    Input is truncated before the call, so size-based 429 errors
    should not occur.
    """
    for attempt in range(max_retries):
        try:
            return call_llm(prepared_text)

        except Exception as e:
            error_msg = str(e)
            is_rate_limit = (
                "rate_limit" in error_msg.lower() or "429" in error_msg
            )

            if is_rate_limit and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                print(f"\n    Rate limit — waiting {wait}s "
                      f"(retry {attempt + 1}/{max_retries})", flush=True)
                time.sleep(wait)
                continue

            if attempt == max_retries - 1:
                print(f"\n    Failed after {max_retries} attempts: "
                      f"{error_msg[:120]}", flush=True)
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
                    "tokens_input":    0,
                    "tokens_output":   0,
                    "tokens_total":    0,
                    "model":           MODEL,
                    "raw_response":    error_msg,
                    "was_truncated":   False,
                    "original_tokens": 0,
                }

            time.sleep(2)

    return call_llm(prepared_text)
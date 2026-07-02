"""Light gpt-4o-mini tie-break: is a transcript fluent speech in the target
language? Mirrors ingest.language_filter.is_in_language (pydantic structured
output, lazy client). Fail-CLOSED (False) on error: a borderline segment that
can't be verified is dropped; the collector simply gathers another."""

from __future__ import annotations

from pydantic import BaseModel

MODEL = "gpt-4o-mini"
_MIN_TRANSCRIPT = 3


class FluencyVerdict(BaseModel):
    is_fluent_target_language: bool
    reason: str


def judge_transcript(
    transcript: str, target_language: str, *, client=None, max_retries: int = 3
) -> bool:
    """True iff `transcript` reads as fluent `target_language` speech."""
    import time
    transcript = (transcript or "").strip()
    if len(transcript) < _MIN_TRANSCRIPT:
        return False
    try:
        from openai import OpenAI
        c = client or OpenAI()
        for attempt in range(max_retries):
            try:
                resp = c.chat.completions.parse(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "You judge whether a speech transcript is FLUENT, natural speech in a "
                            "target language. Set is_fluent_target_language=False for music lyrics, "
                            "noise, gibberish, or text clearly in a DIFFERENT language."
                        )},
                        {"role": "user", "content": (
                            f"Target language: {target_language}.\nTranscript: {transcript!r}\n"
                            "Is this fluent target-language speech?"
                        )},
                    ],
                    response_format=FluencyVerdict,
                )
                verdict = resp.choices[0].message.parsed
                return bool(verdict and verdict.is_fluent_target_language)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        return False
    except Exception:
        return False

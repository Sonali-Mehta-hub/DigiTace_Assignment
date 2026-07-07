"""
Reply-Intent Classifier — powered by Grok (xAI).

Grok's API is OpenAI-compatible, so we use the `openai` SDK pointed at
xAI's base URL instead of OpenAI's.

Approach: single few-shot prompt, not a trained ML model — dataset is
small (60 examples), and intent depends on subtle phrasing, which an
LLM handles better than TF-IDF/logistic regression at this scale.
"""

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env file into environment variables

LABELS = ["interested", "not_interested", "pricing_query", "availability_query", "unclear"]

SYSTEM_PROMPT = """You are an intent classifier for an influencer marketing agency.
A brand reached out to a creator for a paid collaboration. The creator has replied.
Classify the creator's reply into EXACTLY ONE of these 5 categories:

- interested: creator clearly wants to move forward / accepts the collab
- not_interested: creator declines or passes on the collab
- pricing_query: creator's main focus is asking about budget, rate, payment, or compensation
- availability_query: creator's main focus is asking about timeline, deadline, or scheduling
- unclear: reply is vague, sarcastic, off-topic, or too ambiguous to confidently classify

Rules:
- If a message expresses interest AND asks about price/timeline, classify by the PRIMARY
  ask - if dominated by a question about money, pick pricing_query; if dominated by a
  question about dates/scheduling, pick availability_query.
- Short, vague, non-committal replies ("k", "we'll see", "maybe") are unclear, not
  not_interested - not_interested requires an explicit decline.

Respond ONLY with a JSON object, no preamble, no markdown fences:
{"intent": "<one of the 5 labels>", "confidence": <float between 0 and 1>}
"""

FEW_SHOT_EXAMPLES = [
    ("Sure, I'm down. Just send over the content brief whenever you're ready.", "interested"),
    ("hard pass, I don't do sponsored posts for supplement brands anymore", "not_interested"),
    ("What's your rate card look like? Also is this a flat fee or performance based?", "pricing_query"),
    ("when would you need me to film, is there a hard deadline?", "availability_query"),
    ("lol okay we'll see", "unclear"),
]


def _get_client() -> OpenAI:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "XAI_API_KEY not set. Add it to your .env file or export it in your shell."
        )
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def _build_user_prompt(reply_text: str) -> str:
    example_block = "\n".join(
        f'Reply: "{text}"\nAnswer: {{"intent": "{label}", "confidence": 0.95}}'
        for text, label in FEW_SHOT_EXAMPLES
    )
    return (
        f"Here are some labeled examples:\n\n{example_block}\n\n"
        f'Now classify this reply:\nReply: "{reply_text}"\nAnswer:'
    )


def _extract_json(text: str) -> dict:
    """Strip markdown fences if the model adds them, then parse JSON."""
    cleaned = re.sub(r"```json|```", "", text).strip()
    return json.loads(cleaned)


def classify_reply(reply_text: str, client: OpenAI = None) -> dict:
    """
    Classify a single creator reply using Grok.
    Returns: {"intent": str, "confidence": float}
    """
    if client is None:
        client = _get_client()

    response = client.chat.completions.create(
        model="grok-4-fast",  # swap to "grok-4" for higher quality if needed
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(reply_text)},
        ],
        max_tokens=100,
    )

    raw_text = response.choices[0].message.content
    result = _extract_json(raw_text)

    if result.get("intent") not in LABELS:
        raise ValueError(f"Model returned invalid label: {result}")

    return {
        "intent": result["intent"],
        "confidence": float(result.get("confidence", 0.0)),
    }


if __name__ == "__main__":
    # quick smoke test — run this file directly to sanity check your setup
    sample = "What's your rate for a story post?"
    print(f"Testing classifier with: {sample!r}")
    print(classify_reply(sample))
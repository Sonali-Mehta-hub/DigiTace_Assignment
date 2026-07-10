"""
Reply-Intent Classifier — powered by Groq (fast inference, free tier).

Groq's API is OpenAI-compatible, so we use the `openai` SDK pointed at
Groq's base URL instead of OpenAI's.

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

FEW_SHOT_EXAMPLES = [
    ("Sure, I'm down. Just send over the content brief whenever you're ready.", "interested"),
    ("hard pass, I don't do sponsored posts for supplement brands anymore", "not_interested"),
    ("What's your rate card look like? Also is this a flat fee or performance based?", "pricing_query"),
    ("when would you need me to film, is there a hard deadline?", "availability_query"),
    ("lol okay we'll see", "unclear"),
    ("k", "unclear"),
    ("haha maybe, depends", "unclear"),
    ("interesting 👀", "unclear"),
    ("lol this is random but ok", "unclear"),
    ("haha nice, anyway how's your day going", "unclear"),
    ("lol who is this again", "unclear"),
]

FEW_SHOT_EXAMPLES = [
    ("Sure, I'm down. Just send over the content brief whenever you're ready.", "interested"),
    ("hard pass, I don't do sponsored posts for supplement brands anymore", "not_interested"),
    ("What's your rate card look like? Also is this a flat fee or performance based?", "pricing_query"),
    ("when would you need me to film, is there a hard deadline?", "availability_query"),
    ("lol okay we'll see", "unclear"),
    ("k", "unclear"),
    ("haha maybe, depends", "unclear"),
    ("interesting 👀", "unclear"),
    ("lol this is random but ok", "unclear"),
    ("haha nice, anyway how's your day going", "unclear"),
    ("lol who is this again", "unclear"),
]

SYSTEM_PROMPT = (
    "You are an intent classifier for creator outreach replies. "
    "Given a creator's reply, classify it into exactly one of these intents: "
    "interested, not_interested, pricing_query, availability_query, unclear. "
    "Respond ONLY with a JSON object in the form "
    '{"intent": "<one of the labels above>", "confidence": <float between 0 and 1>}. '
    "Do not include any other text, explanation, or markdown formatting."
)


def _get_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file or export it in your shell."
        )
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


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
    Classify a single creator reply using Groq.
    Returns: {"intent": str, "confidence": float}
    """
    if client is None:
        client = _get_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # fast + free-tier friendly; good for simple 5-way classification
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(reply_text)},
        ],
        max_tokens=100,
      temperature=0,
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
    sample = "What's your rate for this collaboration?"
    print(f"Testing the classifier with: {sample!r}")
    print(classify_reply(sample))

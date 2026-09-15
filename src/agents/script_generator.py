"""Agent 3: Script Generator.

Takes classifier tags + trend notes + niche/tone and produces hook options,
a 5-part script, a caption, and hashtags as structured JSON.
"""
import json
import os
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-5"

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "hooks": {
            "type": "array",
            "description": (
                "2 to 3 alternative hook lines, each grabbing attention within 2 seconds"
            ),
            "items": {"type": "string"},
        },
        "script": {
            "type": "object",
            "description": "The full script following the 5-part framework",
            "properties": {
                "hook": {"type": "string", "description": "Chosen hook, first 2 seconds"},
                "context": {"type": "string", "description": "Who/what/why"},
                "value": {"type": "string", "description": "The technique, moment, or insight"},
                "payoff": {"type": "string", "description": "The reveal/result"},
                "cta": {"type": "string", "description": "Book now / follow / save / shop"},
            },
            "required": ["hook", "context", "value", "payoff", "cta"],
            "additionalProperties": False,
        },
        "caption": {"type": "string", "description": "Instagram caption for the post"},
        "hashtags": {
            "type": "array",
            "description": "Relevant hashtags, without the # symbol",
            "items": {"type": "string"},
        },
    },
    "required": ["hooks", "script", "caption", "hashtags"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an expert short-form video scriptwriter specializing in Instagram Reels, \
working across any content niche (makeup, fitness, food, fashion, travel, etc.).

Every script you write follows this 5-part framework:
- Hook: grab attention in the first 2 seconds
- Context: who/what/why, oriented quickly
- Value/Story: the technique, moment, or insight that delivers on the hook
- Payoff: the reveal/result
- CTA: book now / follow / save / shop - pick whichever fits the content and niche

Base every line on the actual input you're given - the niche, the classifier tags (content \
type, mood, key visual moment), and the trend notes. Pull concrete details straight out of \
that input (the specific moment, technique, ingredient, location, before/after, whatever it \
names) instead of inventing generic scenarios. If the input is thin, stay minimal rather than \
padding with invented specifics. Treat trend notes as inspiration for pacing/format only, \
never content to copy verbatim.

Every hook, the script, and the caption must be eye-catching, concise, and unique:
- Eye-catching: a sharp specific detail or unexpected angle pulled from the input, not a \
generic tease. Avoid cliche openers ("POV:", "Wait for it", "You won't believe").
- Concise: every line earns its place. Cut any word, clause, or sentence that doesn't add new \
information - say it once, well, and move on. Favor short punchy lines over run-ons.
- Unique: it must read as built for this specific piece of content, not a template with the \
niche swapped in. If a line could paste onto a different creator's content unchanged, rewrite \
it until it can't.

Produce 2-3 distinct hook options (different angles), then a full script built around the \
strongest hook, a ready-to-post caption, and a relevant hashtag set."""


def _build_user_prompt(
    niche: str,
    tags: Optional[dict[str, Any]],
    trend_notes: str,
    tone: str,
) -> str:
    parts = [f"Niche: {niche}"]
    if tone:
        parts.append(f"Tone/voice: {tone}")
    if tags:
        parts.append(f"Content tags (from vision classifier):\n{json.dumps(tags, indent=2)}")
    if trend_notes:
        parts.append(f"Trend/reference notes:\n{trend_notes}")
    parts.append(
        "Generate the hook options, full 5-part script, caption, and hashtags for this reel."
    )
    return "\n\n".join(parts)


def generate_script(
    niche: str,
    tags: Optional[dict[str, Any]] = None,
    trend_notes: str = "",
    tone: str = "",
    client: Optional[anthropic.Anthropic] = None,
    model: str = MODEL,
) -> dict[str, Any]:
    """Generate a reel script.

    Args:
        niche: content niche, e.g. "makeup", "fitness", "food".
        tags: structured tags from Agent 1 (Classifier), if available.
        trend_notes: trend/reference notes from Agent 2, if available.
        tone: creator's tone/voice notes, e.g. "playful, fast-paced, Gen Z".
        client: optional pre-built anthropic.Anthropic client (for reuse/testing).
        model: Claude model id to use.

    Returns:
        dict matching SCRIPT_SCHEMA: hooks, script, caption, hashtags.
    """
    if not niche:
        raise ValueError("niche is required")

    client = client or anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": SCRIPT_SCHEMA}},
        messages=[
            {"role": "user", "content": _build_user_prompt(niche, tags, trend_notes, tone)}
        ],
    )

    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def refine_script(
    previous_script: dict[str, Any],
    feedback: str,
    niche: str,
    tags: Optional[dict[str, Any]] = None,
    trend_notes: str = "",
    tone: str = "",
    client: Optional[anthropic.Anthropic] = None,
    model: str = MODEL,
) -> dict[str, Any]:
    """Revise a previously generated script based on user feedback.

    Args:
        previous_script: prior output from generate_script/refine_script (SCRIPT_SCHEMA shape).
        feedback: the user's description of what to change.
        niche, tags, trend_notes, tone: same context used to generate the original script.
        client: optional pre-built anthropic.Anthropic client (for reuse/testing).
        model: Claude model id to use.

    Returns:
        dict matching SCRIPT_SCHEMA: a revised version of the script.
    """
    if not feedback:
        raise ValueError("feedback is required")

    client = client or anthropic.Anthropic()

    prompt = (
        f"{_build_user_prompt(niche, tags, trend_notes, tone)}\n\n"
        f"Here's the previous output:\n{json.dumps(previous_script, indent=2)}\n\n"
        f"The user wants this changed: {feedback}\n\n"
        "Generate a revised version. Only change what the feedback asks for - keep everything "
        "else from the previous output as-is."
    )

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": SCRIPT_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )

    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path

    result = generate_script(
        niche="makeup",
        tags={
            "content_type": "tutorial",
            "mood": "energetic",
            "key_visual_moment": "before/after reveal of a smokey eye look",
        },
        trend_notes=(
            "Trending format this month: fast jump-cut tutorials with a text overlay "
            "counting down steps, ending on a slow-motion reveal."
        ),
        tone="playful, confident, talks directly to camera",
    )
    print(json.dumps(result, indent=2))

    outputs_dir = Path(__file__).resolve().parents[2] / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")

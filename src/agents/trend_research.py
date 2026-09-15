"""Agent 2: Trend + Reference Research.

Takes a niche (and optionally a sample reel) and returns current trending
reel hooks/formats via Claude's web search tool, plus an optional structural
pacing breakdown of the sample reel. Output feeds into Agent 3 (Script
Generator) as its `trend_notes` argument.

Risk: no live Instagram API access - treat results as inspiration, not
ground truth. A sample reel's structure (cuts, text style, reveal timing) is
analyzed, never its actual content/audio/captions.
"""
import json
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv

try:
    from agents.classifier import load_media_blocks
except ImportError:
    from classifier import load_media_blocks

load_dotenv()

MODEL = "claude-opus-5"

TREND_SCHEMA = {
    "type": "object",
    "properties": {
        "trend_notes": {
            "type": "string",
            "description": (
                "Current trending reel hooks/formats for this niche, based on web search. "
                "Inspiration only, not ground truth."
            ),
        },
        "pattern_breakdown": {
            "type": "string",
            "description": (
                "Structural pacing breakdown of the sample reel (cuts, text style, reveal "
                "timing) if one was provided - empty string if no sample reel was given."
            ),
        },
    },
    "required": ["trend_notes", "pattern_breakdown"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a trend researcher for a short-form video (Instagram Reels) \
pipeline that works across any creator niche (makeup, fitness, food, fashion, travel, etc.).

Use web search to find CURRENT (this month) trending reel hooks and formats for the given \
niche - hook styles, pacing patterns, text-overlay conventions, transition styles. You have \
no live Instagram API access, so treat everything you find as inspiration, not ground truth; \
note in trend_notes if results feel thin or dated. Never suggest copying a specific creator's \
actual content, audio, or captions - only structural/format patterns.

If a sample reel is attached, analyze ONLY its structural pattern: cut frequency, text style, \
reveal timing, pacing rhythm. Never describe or reuse its actual content, audio, or captions - \
structure only. If no sample reel is attached, leave pattern_breakdown as an empty string.

Keep trend_notes concise and actionable - a scriptwriter will use it directly, not a reader \
browsing trend reports."""


def research_trends(
    niche: str,
    sample_reel_path: Optional[str] = None,
    client: Optional[anthropic.Anthropic] = None,
    model: str = MODEL,
) -> dict[str, Any]:
    """Research current trending reel hooks/formats for a niche.

    Args:
        niche: content niche, e.g. "makeup", "fitness", "food".
        sample_reel_path: optional path to a sample reel (photo or video) whose
            structural pacing/format should be broken down.
        client: optional pre-built anthropic.Anthropic client (for reuse/testing).
        model: Claude model id to use.

    Returns:
        dict matching TREND_SCHEMA: trend_notes, pattern_breakdown.
    """
    if not niche:
        raise ValueError("niche is required")

    client = client or anthropic.Anthropic()

    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": f"Niche: {niche}\n\nFind current trending reel hooks/formats for this niche.",
    }]
    if sample_reel_path:
        content.append({
            "type": "text",
            "text": "A sample reel is attached below - break down its structural pacing pattern.",
        })
        content.extend(load_media_blocks(sample_reel_path))

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
        output_config={"format": {"type": "json_schema", "schema": TREND_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )

    text_blocks = [block.text for block in response.content if block.type == "text"]
    if not text_blocks:
        raise RuntimeError("No text content in trend research response")
    return json.loads(text_blocks[-1])


if __name__ == "__main__":
    import sys
    from datetime import datetime
    from pathlib import Path

    niche = sys.argv[1] if len(sys.argv) > 1 else "makeup"
    sample_reel = sys.argv[2] if len(sys.argv) > 2 else None

    result = research_trends(niche, sample_reel_path=sample_reel)
    print(json.dumps(result, indent=2))

    outputs_dir = Path(__file__).resolve().parents[2] / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / f"trends_{niche}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")

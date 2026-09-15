"""Agent 1: Classifier.

Takes a creator's raw photos and/or videos and uses Claude's vision API to tag
the content: niche, content type, mood, and the single most reel-worthy visual
moment. Videos are reduced to 3 key frames (start, middle, end) via OpenCV
before classification, so they go through the same vision path as photos.
Output is structured JSON, not prose - feeds directly into Agent 3 (Script
Generator) as its `tags` argument.
"""
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Optional

import anthropic
import cv2
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-5"
NUM_VIDEO_FRAMES = 3

SUPPORTED_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

SUPPORTED_VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

TAGS_SCHEMA = {
    "type": "object",
    "properties": {
        "niche": {
            "type": "string",
            "description": (
                "The content niche these photos belong to, e.g. makeup, fitness, food, "
                "fashion, travel. Be specific to what's actually shown, not a generic guess."
            ),
        },
        "content_type": {
            "type": "string",
            "description": (
                "The reel format these photos suggest, e.g. tutorial, transformation, "
                "day-in-life, product reveal, recipe reveal, before/after."
            ),
        },
        "mood": {
            "type": "string",
            "description": "The mood/energy conveyed by the photos, e.g. energetic, cozy, raw, playful.",
        },
        "key_visual_moment": {
            "type": "string",
            "description": (
                "The single most reel-worthy moment or frame across the provided photos - "
                "the shot the whole reel should be built around."
            ),
        },
    },
    "required": ["niche", "content_type", "mood", "key_visual_moment"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a content classifier for a short-form video (Instagram Reels) \
pipeline that works across any creator niche (makeup, fitness, food, fashion, travel, etc.).

You are given raw photos and/or video key frames from a creator - not a finished reel, not \
curated content. Some images are labeled as start/middle/end frames from the same source \
video; treat those as a sequence, not independent photos. Your job is to look at what's \
actually shown and produce structured tags that a downstream scriptwriting agent will use to \
write a reel script.

Rules:
- Base every tag on what is visually present - never guess at context the images don't show.
- Be specific, not generic. "Makeup" is not a mood. "Energetic" is not a niche. Each field \
should reflect this particular set of media, not a category description.
- key_visual_moment must point to one concrete moment/frame (not a vague summary) - the shot \
with the most reel potential, e.g. a reveal, a peak action, a close-up detail, a punchline.
- If the media spans multiple moments (e.g. before/after, step-by-step, a video's start vs. \
end frame), consider the whole set before choosing content_type and key_visual_moment.
- Output only the structured tags - no prose, no explanation outside the schema."""


def _image_block(data: bytes, media_type: str) -> dict[str, Any]:
    encoded = base64.standard_b64encode(data).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": encoded},
    }


def _extract_video_frames(path: str, num_frames: int = NUM_VIDEO_FRAMES) -> list[bytes]:
    """Extract up to `num_frames` frames (start, middle, end) from a video as JPEG bytes."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Video has no readable frames: {path}")

        last_index = frame_count - 1
        step = last_index / (num_frames - 1) if num_frames > 1 else 0
        indices = sorted({round(step * i) for i in range(num_frames)})

        frames: list[bytes] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frames.append(buffer.tobytes())

        if not frames:
            raise ValueError(f"Could not extract any frames from video: {path}")
        return frames
    finally:
        cap.release()


def load_media_blocks(path: str) -> list[dict[str, Any]]:
    """Build Claude vision content blocks for one photo or video path.

    Videos are expanded into labeled start/middle/end frame blocks; reused by
    other agents (e.g. trend research's sample-reel breakdown) that also need
    to hand media to Claude's vision API.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = file_path.suffix.lower()

    if ext in SUPPORTED_VIDEO_TYPES:
        frames = _extract_video_frames(str(file_path))
        labels = ["start", "middle", "end"][: len(frames)]
        blocks: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"The following {len(frames)} image(s) are key frames "
                f"({', '.join(labels)}) from video: {file_path.name}"
            ),
        }]
        blocks.extend(_image_block(frame, "image/jpeg") for frame in frames)
        return blocks

    media_type = SUPPORTED_IMAGE_TYPES.get(ext) or mimetypes.guess_type(file_path.name)[0]
    if media_type not in SUPPORTED_IMAGE_TYPES.values():
        raise ValueError(
            f"Unsupported file type for {path}: {ext or 'unknown'}. "
            f"Supported images: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}; "
            f"supported videos: {', '.join(sorted(SUPPORTED_VIDEO_TYPES))}"
        )
    return [_image_block(file_path.read_bytes(), media_type)]


def classify_content(
    media_paths: list[str],
    client: Optional[anthropic.Anthropic] = None,
    model: str = MODEL,
) -> dict[str, Any]:
    """Classify a creator's raw photos and/or videos into structured tags.

    Args:
        media_paths: file paths to the creator's raw photos (jpg/jpeg/png/gif/webp) and/or
            videos (mp4/mov/avi/mkv/webm). Videos are reduced to 3 key frames
            (start/middle/end) before classification.
        client: optional pre-built anthropic.Anthropic client (for reuse/testing).
        model: Claude model id to use.

    Returns:
        dict matching TAGS_SCHEMA: niche, content_type, mood, key_visual_moment.
    """
    if not media_paths:
        raise ValueError("media_paths is required and must contain at least one path")

    client = client or anthropic.Anthropic()

    content: list[dict[str, Any]] = []
    for path in media_paths:
        content.extend(load_media_blocks(path))
    content.append({
        "type": "text",
        "text": (
            f"The above is raw media from a creator (from {len(media_paths)} source file(s)), "
            "meant to become one Instagram Reel. Classify it."
        ),
    })

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": TAGS_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )

    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


if __name__ == "__main__":
    import sys
    from datetime import datetime

    if len(sys.argv) < 2:
        print("Usage: python src/agents/classifier.py <photo_or_video1> [photo_or_video2 ...]")
        print("Provide one or more real photo/video file paths to classify.")
        sys.exit(1)

    result = classify_content(sys.argv[1:])
    print(json.dumps(result, indent=2))

    outputs_dir = Path(__file__).resolve().parents[2] / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / f"tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")

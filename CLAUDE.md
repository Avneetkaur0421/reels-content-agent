# Reels Content Agent — Full Project

## Goal
An agentic pipeline that takes a creator's raw photos/videos and produces 
a ready-to-post Instagram reel — script, caption, hashtags, and eventually 
the rendered video itself. Works for any niche (makeup, fitness, food, 
fashion, travel, etc.), not tied to one influencer type.

## Build order (build & test each agent standalone before wiring together)
1. Agent 3 — Script Generator (start here, most reliable)
2. Agent 4 — Video Assembler
3. Agent 1 — Classifier
4. Agent 2 — Trend + Reference Research
5. Bot/intake + orchestration (wire everything together last)

---

## Agent 1: Classifier
**Input:** raw photos/videos
**Job:** Claude vision tags content type (niche, mood, key visual moment)
**Output:** structured tags (JSON), not prose
**Tech:** Claude API (vision)

## Agent 2: Trend + Reference Research
**Input:** niche/tags from Agent 1, optional sample reel upload
**Job:** 
- Web search for trending reel hooks/formats for that niche, current month
- If sample reel provided: Claude vision breaks down pacing/structure 
  (cuts, text style, reveal timing) — structural pattern only, never 
  copies actual content/audio/captions
**Output:** trend notes + optional pattern breakdown
**Tech:** Claude API + web search tool
**Risk:** treat as inspiration, not ground truth — no live Instagram API access

## Agent 3: Script Generator
**Input:** Agent 1 tags + Agent 2 trend notes + niche + tone/voice notes
**Job:** generates 2-3 hook options, full script, caption, hashtags using 
the 5-part framework below
**Output:** structured script (JSON or markdown)
**Tech:** Claude API (text)

### 5-Part Framework (core system prompt)
- Hook: grab attention in 2 seconds
- Context: who/what/why
- Value/Story: the technique, moment, or insight
- Payoff: the reveal/result
- CTA: book now / follow / save / shop

## Agent 4: Video Assembler
**Input:** script from Agent 3 + original photos
**Job:** fills a pre-built Canva (Autofill API) or Creatomate template 
matching the content type, renders MP4
**Output:** rendered video file
**Tech:** Canva Connect API or Creatomate API
**Risk:** need a small library of templates per content style (tutorial, 
transformation, day-in-life) — template design still needs a human touch 
periodically

---

## Orchestration
`src/pipeline.py` runs: Classifier → Trend Research → Script Generator → 
Video Assembler, in sequence, passing output forward.

## Intake (build last)
`src/bot/intake.py` — WhatsApp/Telegram webhook or shared folder watcher 
that triggers the pipeline when new content arrives.

## Tech stack
- Python (venv: `reels/`, already created)
- `anthropic` Python SDK (Claude API — text + vision)
- `python-dotenv` for env vars
- Web search (via Claude API's web search tool)
- Canva Connect API or Creatomate API (video rendering) — via `requests`
- `.env` for API keys: ANTHROPIC_API_KEY, CANVA_API_KEY / CREATOMATE_API_KEY

## Folder structure
Reels_creation/
├── reels/                  (venv — don't touch)
├── .env
├── requirements.txt
├── CLAUDE.md
├── src/
│   ├── bot/intake.py
│   ├── agents/
│   │   ├── classifier.py
│   │   ├── trend_research.py
│   │   ├── script_generator.py
│   │   └── video_assembler.py
│   ├── templates/
│   ├── pipeline.py
│   └── config.py
├── outputs/
└── README.md
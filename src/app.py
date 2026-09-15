"""Streamlit UI for the Reel Content Agent pipeline.

Upload raw photos/videos, pick a niche, and run Classifier -> Trend Research
-> Script Generator in sequence, passing each agent's output to the next.
"""
import sys
import tempfile
from pathlib import Path

import anthropic
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.classifier import classify_content  # noqa: E402
from agents.script_generator import generate_script, refine_script  # noqa: E402
from agents.trend_research import research_trends  # noqa: E402

NICHE_OPTIONS = ["Beauty", "Fitness", "Food", "Fashion", "Travel"]
IMAGE_EXTS = {".jpg", ".png"}
VIDEO_EXTS = {".mp4", ".mov"}


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


if "niche" not in st.session_state:
    st.session_state.niche = None
if "result" not in st.session_state:
    st.session_state.result = None
if "tags" not in st.session_state:
    st.session_state.tags = None
if "trend_notes" not in st.session_state:
    st.session_state.trend_notes = ""

st.set_page_config(page_title="Reel content agent", page_icon="🎬")

st.title("Reel content agent")
st.caption("Turn raw photos and videos into a ready-to-post reel script.")

uploaded_files = st.file_uploader(
    "Upload photos and videos",
    type=["jpg", "png", "mp4", "mov"],
    accept_multiple_files=True,
)

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 5))
    for i, f in enumerate(uploaded_files):
        ext = Path(f.name).suffix.lower()
        is_video = ext in VIDEO_EXTS
        with cols[i % len(cols)]:
            if is_video:
                st.video(f)
            else:
                st.image(f, use_container_width=True)
            st.caption(f"{'🎬' if is_video else '📷'} {f.name}")

st.write("Niche")
chip_cols = st.columns(len(NICHE_OPTIONS))
for col, option in zip(chip_cols, NICHE_OPTIONS):
    with col:
        is_selected = st.session_state.niche == option
        if st.button(
            option,
            key=f"niche_{option}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):
            st.session_state.niche = option

run_disabled = not uploaded_files or not st.session_state.niche
run_clicked = st.button("Run pipeline", type="primary", disabled=run_disabled)

if run_clicked:
    client = get_client()

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for f in uploaded_files:
            path = Path(tmp_dir) / f.name
            path.write_bytes(f.getvalue())
            paths.append(str(path))

        with st.status("Classifying content", expanded=True) as status:
            try:
                tags = classify_content(paths, client=client)
                status.write("Content classified.")

                status.update(label="Researching trends", state="running")
                trends = research_trends(st.session_state.niche, client=client)
                status.write("Trend notes gathered.")

                status.update(label="Writing script", state="running")
                script = generate_script(
                    niche=st.session_state.niche,
                    tags=tags,
                    trend_notes=trends["trend_notes"],
                    client=client,
                )
                status.update(label="Pipeline complete", state="complete")
            except Exception as e:
                status.update(label="Pipeline failed", state="error")
                st.error(str(e))
                script = None
                tags = None
                trends = {"trend_notes": ""}

    st.session_state.result = script
    st.session_state.tags = tags
    st.session_state.trend_notes = trends["trend_notes"]

if st.session_state.result:
    script = st.session_state.result

    tab_hook, tab_script, tab_caption = st.tabs(["Hook", "Script", "Caption"])

    with tab_hook:
        for hook in script["hooks"]:
            st.write(f"- {hook}")

    with tab_script:
        s = script["script"]
        st.markdown(f"**Hook:** {s['hook']}")
        st.markdown(f"**Context:** {s['context']}")
        st.markdown(f"**Value/Story:** {s['value']}")
        st.markdown(f"**Payoff:** {s['payoff']}")
        st.markdown(f"**CTA:** {s['cta']}")

    with tab_caption:
        st.write(script["caption"])

    if st.button("Copy hashtags"):
        st.code(" ".join(f"#{tag}" for tag in script["hashtags"]), language=None)

    st.divider()
    feedback = st.text_area("Want changes? Describe what to fix")

    refine_col, regenerate_col = st.columns([3, 1])
    with refine_col:
        refine_clicked = st.button(
            "Refine", type="primary", disabled=not feedback, use_container_width=True
        )
    with regenerate_col:
        regenerate_clicked = st.button("Regenerate from scratch", use_container_width=True)

    if refine_clicked:
        with st.spinner("Refining script..."):
            try:
                st.session_state.result = refine_script(
                    previous_script=st.session_state.result,
                    feedback=feedback,
                    niche=st.session_state.niche,
                    tags=st.session_state.tags,
                    trend_notes=st.session_state.trend_notes,
                    client=get_client(),
                )
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if regenerate_clicked:
        with st.spinner("Regenerating script..."):
            try:
                st.session_state.result = generate_script(
                    niche=st.session_state.niche,
                    tags=st.session_state.tags,
                    trend_notes=st.session_state.trend_notes,
                    client=get_client(),
                )
                st.rerun()
            except Exception as e:
                st.error(str(e))

import streamlit as st
import torch
from model import generate_creative_story, load_checkpoint, train_model
from video_gen import generate_story_video

st.set_page_config(page_title="Horror & Mystery Tiny Transformer", page_icon="🕯️", layout="centered")

st.title("🕯️ Horror & Mystery Story & Animated Video Generator")
st.caption("Tiny Transformer Story Engine + Story-Matched AI Scene Animation + Voice Narration")

with st.sidebar:
    st.header("1. Story Engine Mode")
    engine_mode = st.radio(
        "Choose Transformer Engine",
        [
            "Custom Prompt Mode (SmolLM2-135M Tiny Transformer)",
            "From-Scratch PyTorch Tiny Transformer",
        ],
    )

    if engine_mode == "From-Scratch PyTorch Tiny Transformer":
        steps = st.slider("Training Steps", min_value=100, max_value=1500, value=500, step=100)
        if st.button("Train / Retrain From-Scratch Model"):
            with st.spinner("Training PyTorch Tiny Transformer..."):
                losses = train_model(steps=steps)
            st.success(f"Trained! Final loss: {losses[-1]:.4f}")
            st.line_chart(losses)
    else:
        steps = 500

    st.header("2. Generation & Video Controls")
    temperature = st.slider("Temperature (Creativity)", min_value=0.2, max_value=1.4, value=0.85, step=0.05)
    max_tokens = st.slider("Max Generated Tokens / Chars", min_value=80, max_value=400, value=180, step=20)
    add_narration = st.checkbox("Include Voice Narration Audio (gTTS)", value=True)
    hf_token = st.text_input(
        "Optional Hugging Face Token (for FLUX scene art)",
        type="password",
        help="Leave blank to use the free built-in AI scene art generator automatically.",
    )

starter_prompts = [
    "A crow and an evil panda meet in a foggy bamboo forest at midnight",
    "A candlelit wedding ceremony inside an abandoned cathedral at midnight",
    "The grandfather clock in the hallway struck thirteen",
    "Inspector Hale examined the sealed envelope",
]
selected_starter = st.selectbox("Quick Starter Prompt", ["Custom"] + starter_prompts)
default_prompt = "" if selected_starter == "Custom" else selected_starter

prompt = st.text_area(
    "Enter any horror or mystery idea or opening:",
    value=default_prompt or "A crow and an evil panda guard an abandoned temple in the mist.",
    height=95,
)

if "generated_story" not in st.session_state:
    st.session_state.generated_story = ""

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Story", type="primary", use_container_width=True):
        if engine_mode == "Custom Prompt Mode (SmolLM2-135M Tiny Transformer)":
            with st.spinner("Generating story with SmolLM2-135M Tiny Transformer..."):
                st.session_state.generated_story = generate_creative_story(
                    prompt, max_new_tokens=max_tokens, temperature=temperature
                )
        else:
            loaded = load_checkpoint()
            if loaded is None:
                with st.spinner("First run: training PyTorch Tiny Transformer..."):
                    train_model(steps=steps)
                    loaded = load_checkpoint()

            model, tokenizer, _, device = loaded
            encoded = tokenizer.encode(prompt)
            if not encoded:
                st.warning("Use standard English letters/punctuation matching the training text.")
            else:
                ctx = torch.tensor([encoded], dtype=torch.long, device=device)
                out_ids = model.generate(ctx, max_new_tokens=max_tokens, temperature=temperature)[0].tolist()
                st.session_state.generated_story = tokenizer.decode(out_ids)

with col2:
    if st.button("Generate Animated Video (.mp4)", use_container_width=True):
        if not st.session_state.generated_story.strip():
            st.warning("Generate a story first, then click Generate Animated Video.")
        else:
            with st.spinner("Generating story-matched scene art, camera motion & voice narration..."):
                video_path = generate_story_video(
                    st.session_state.generated_story,
                    output_path="story_scene.mp4",
                    add_narration=add_narration,
                    hf_token=hf_token,
                )
            st.session_state.video_path = video_path

if st.session_state.generated_story:
    st.subheader("Generated Story")
    st.write(st.session_state.generated_story)

if "video_path" in st.session_state and st.session_state.video_path:
    st.subheader("Generated Animated Story Video")
    st.video(st.session_state.video_path)
    with open(st.session_state.video_path, "rb") as vf:
        st.download_button("Download MP4 Video", data=vf, file_name="horror_mystery_story.mp4", mime="video/mp4")

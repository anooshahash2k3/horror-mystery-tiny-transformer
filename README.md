# AI-Based Horror and Mystery Story & Animated Video Generator

A semester project for **Advanced Topics in Generative AI** built with a **Tiny Transformer** story generator (`SmolLM2-135M` + custom from-scratch PyTorch Tiny Transformer), story-matched AI scene illustration + motion video rendering, `gTTS` voice narration, and an interactive **Streamlit** web interface.

## Project Structure
- `app.py` — Streamlit web application for story generation, training, and animated `.mp4` video export.
- `model.py` — Custom PyTorch Tiny Transformer (`CharTokenizer` + causal `TinyTransformer`) and `SmolLM2-135M` custom-prompt story engine.
- `video_gen.py` — Story-matched AI scene artwork generator with Ken Burns motion, subtitles, and `gTTS` audio narration.
- `sample_stories.txt` — Starter horror and mystery corpus for the from-scratch PyTorch Tiny Transformer.
- `requirements.txt` — Python dependencies.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

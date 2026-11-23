import io
import math
import os
import random
import tempfile
import textwrap
import numpy as np
import requests
from gtts import gTTS
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from moviepy.editor import AudioFileClip, VideoClip, concatenate_videoclips


def _split_scenes(story_text: str, max_scenes: int = 4):
    sentences = [s.strip() for s in story_text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return [story_text.strip() or "A shadow moved across the hallway."]
    return [s + "." for s in sentences[:max_scenes]]


def _get_font(size: int):
    candidates = [
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _generate_scene_artwork(scene_text: str, full_story: str, hf_token: str = "", width: int = 854, height: int = 480):
    prompt = (
        f"3D animated movie still, cinematic dark mystery and spooky atmosphere, rich detailed environment, "
        f"expressive characters, volumetric lighting, Pixar and Tim Burton inspired style: {scene_text} "
        f"Context: {full_story[:140]}"
    )

    if hf_token and hf_token.strip():
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=hf_token.strip())
            img = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-schnell",
            )
            return img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), "HF AI Art"
        except Exception:
            pass

    try:
        import urllib.parse

        encoded_prompt = urllib.parse.quote(prompt[:360])
        seed = abs(hash(scene_text)) % 99999
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={seed}&nologo=true"
        resp = requests.get(url, timeout=28)
        if resp.status_code == 200 and len(resp.content) > 2000:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return img.resize((width, height), Image.Resampling.LANCZOS), "AI Scene Art"
    except Exception:
        pass

    return None, "Fallback"


def _build_procedural_scene_bg(scene_text: str, scene_idx: int, width: int = 854, height: int = 480):
    low = scene_text.lower()
    rng = random.Random(scene_idx * 131 + len(scene_text))
    img = Image.new("RGB", (width, height), (14, 16, 26))
    draw = ImageDraw.Draw(img)

    top_col = (28, 22, 42) if ("candle" in low or "flower" in low or "ceremony" in low) else (12, 16, 28)
    bot_col = (48, 32, 38) if ("candle" in low or "flower" in low) else (22, 28, 38)
    for y in range(height):
        r = int(top_col[0] + (bot_col[0] - top_col[0]) * (y / height))
        g = int(top_col[1] + (bot_col[1] - top_col[1]) * (y / height))
        b = int(top_col[2] + (bot_col[2] - top_col[2]) * (y / height))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    draw.ellipse((width - 220, 45, width - 110, 155), fill=(235, 220, 185))

    for i in range(9):
        tx = i * 110 + rng.randint(-15, 15)
        th = rng.randint(140, 260)
        draw.polygon([(tx, height), (tx + 45, height - th), (tx + 90, height)], fill=(10, 12, 18))

    if "candle" in low or "flower" in low or "aisle" in low:
        for cx in range(120, width - 100, 95):
            draw.rectangle((cx - 5, height - 135, cx + 5, height - 105), fill=(235, 225, 200))
            draw.ellipse((cx - 9, height - 152, cx + 9, height - 132), fill=(255, 180, 60))

    return img


def generate_story_video(
    story_text: str,
    output_path: str = "story_scene.mp4",
    add_narration: bool = True,
    hf_token: str = "",
) -> str:
    scenes = _split_scenes(story_text)
    font_header = _get_font(17)
    font_body = _get_font(22)
    width, height = 854, 480

    clips = []
    temp_audio_files = []

    for idx, scene in enumerate(scenes):
        duration = max(3.5, min(7.5, len(scene) * 0.068))
        audio_clip = None

        if add_narration:
            try:
                tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                tts = gTTS(text=scene, lang="en", slow=False)
                tts.save(tmp_mp3)
                temp_audio_files.append(tmp_mp3)
                audio_clip = AudioFileClip(tmp_mp3)
                duration = max(duration, audio_clip.duration + 0.35)
            except Exception:
                audio_clip = None

        base_img, source_mode = _generate_scene_artwork(scene, story_text, hf_token=hf_token, width=width, height=height)
        if base_img is None:
            base_img = _build_procedural_scene_bg(scene, idx, width=width, height=height)

        canvas_w, canvas_h = int(width * 1.14), int(height * 1.14)
        base_large = base_img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

        wrapped_lines = textwrap.fill(scene, width=54).split("\n")
        pan_dir = 1 if idx % 2 == 0 else -1

        def make_frame(
            t,
            s_idx=idx,
            total=len(scenes),
            dur=duration,
            img_large=base_large,
            lines=wrapped_lines,
            p_dir=pan_dir,
            mode=source_mode,
        ):
            progress = min(1.0, t / max(dur, 0.1))

            zoom = 1.0 + 0.08 * progress
            crop_w = int(width / zoom)
            crop_h = int(height / zoom)
            max_dx = canvas_w - crop_w
            max_dy = canvas_h - crop_h

            if p_dir > 0:
                x0 = int(max_dx * progress)
            else:
                x0 = int(max_dx * (1.0 - progress))
            y0 = int(max_dy * (0.3 + 0.4 * math.sin(progress * math.pi)))

            frame_img = img_large.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(
                (width, height), Image.Resampling.LANCZOS
            )

            flicker = 0.96 + 0.04 * math.sin(t * 7.0)
            frame_img = ImageEnhance.Brightness(frame_img).enhance(flicker)

            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(overlay)

            ov_draw.rectangle((18, 16, 360, 46), fill=(12, 13, 20, 175))
            ov_draw.text(
                (28, 22),
                f"SCENE {s_idx + 1}/{total}  |  {mode.upper()}",
                fill=(245, 110, 115, 255),
                font=font_header,
            )

            box_h = 34 * len(lines) + 28
            box_y0 = height - box_h - 18
            ov_draw.rectangle((24, box_y0, width - 24, height - 18), fill=(10, 11, 18, 195), outline=(175, 55, 65, 220), width=2)

            full_str = "\n".join(lines)
            visible_chars = max(1, int(len(full_str) * min(1.0, progress * 1.4)))
            revealed_lines = full_str[:visible_chars].split("\n")

            y_text = box_y0 + 14
            for line in revealed_lines:
                ov_draw.text((42, y_text), line, fill=(248, 248, 252, 255), font=font_body)
                y_text += 32

            composed = Image.alpha_composite(frame_img.convert("RGBA"), overlay).convert("RGB")
            return np.array(composed)

        clip = VideoClip(make_frame, duration=duration).fadein(0.35).fadeout(0.35)
        if audio_clip is not None:
            clip = clip.set_audio(audio_clip)
        clips.append(clip)

    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac" if add_narration else None,
        audio=add_narration,
        verbose=False,
        logger=None,
    )
    final_video.close()
    for c in clips:
        if c.audio is not None:
            c.audio.close()
        c.close()
    for p in temp_audio_files:
        if os.path.exists(p):
            os.remove(p)

    return os.path.abspath(output_path)

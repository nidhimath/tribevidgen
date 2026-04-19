"""
NeuroAdGen Gradio UI.

Simple interface: type a prompt, get back a video + TribeV2 brain reward scores.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import torch
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Lazy-load models once on first request
# ---------------------------------------------------------------------------
_generator = None
_tribe_reward = None
_models_lock = threading.Lock()
_tribe_lock = threading.Lock()


def _load_models():
    """Load local CogVideoX generator (only needed for local mode)."""
    global _generator
    with _models_lock:
        if _generator is not None:
            return
        from neuroadgen.models.video_gen import VideoGenerator
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        logger.info("Loading VideoGenerator on %s...", device)
        _generator = VideoGenerator(
            model_id=CFG["video_model"]["name"],
            dtype=dtype,
            device=device,
            cache_dir=CFG["paths"]["cache_dir"],
            enable_gradient_checkpointing=False,
            truncated_bptt_k=CFG["reward_opt"]["truncated_bptt_k"],
        )
        logger.info("VideoGenerator ready.")


def _load_tribe():
    """Load TribeV2 reward model (used in both API and local modes)."""
    global _tribe_reward
    with _tribe_lock:
        if _tribe_reward is not None:
            return
        from neuroadgen.models.tribe_reward import TribeReward
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading TribeV2 reward model...")
        _tribe_reward = TribeReward(
            roi_config=CFG["rois"],
            differentiability_strategy="reinforce",
            cache_folder=CFG["tribe"]["cache_folder"],
            hemodynamic_lag_sec=CFG["tribe"]["hemodynamic_lag_sec"],
            prediction_end_sec=CFG["tribe"]["prediction_end_sec"],
            fps=CFG["video_model"]["fps"],
            device=device,
        )
        logger.info("TribeV2 ready.")


# ---------------------------------------------------------------------------
# Video export helper
# ---------------------------------------------------------------------------

def _save_video(video: torch.Tensor, fps: int = 8) -> str:
    import cv2
    import numpy as np
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    T, C, H, W = video.shape
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frames = (video.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return tmp.name


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_pipeline(prompt: str, guidance_scale: float, use_api: bool):
    """
    Generate a video from prompt and score with TribeV2.
    API mode: fal.ai generates video (~30-60s), TribeV2 scores locally.
    Local mode: CogVideoX runs on-device (15-40 min on Mac).
    """
    if not prompt.strip():
        yield None, "Please enter a prompt.", None
        return

    # ------------------------------------------------------------------
    # Step 1: Generate video
    # ------------------------------------------------------------------
    if use_api:
        yield None, "🎬 Generating video via fal.ai API (~30-60s)...", None
        try:
            from neuroadgen.models.video_gen import CogVideoXAPIGenerator
            api_gen = CogVideoXAPIGenerator()
            video_path = api_gen.generate(
                prompt=prompt,
                num_inference_steps=CFG["video_model"]["inference_steps"],
                guidance_scale=guidance_scale,
                num_frames=CFG["video_model"]["duration_frames"],
                fps=CFG["video_model"]["fps"],
            )
        except Exception as e:
            yield None, f"API generation failed: {e}\n\nSet FAL_KEY env var and try again.", None
            return
    else:
        yield None, "⏳ Loading local model (this may take a few minutes)...", None
        _load_models()
        device = next(_generator.pipe.transformer.parameters()).device
        fps = CFG["video_model"]["fps"]
        yield None, "🎬 Generating video locally (~15-40 min on Mac)...", None
        with torch.no_grad():
            video, _ = _generator.generate(
                prompt=prompt,
                num_inference_steps=CFG["video_model"]["inference_steps"],
                guidance_scale=guidance_scale,
                height=CFG["video_model"]["resolution"],
                width=int(CFG["video_model"]["resolution"] * 16 / 9),
                num_frames=CFG["video_model"]["duration_frames"],
            )
        video_path = _save_video(video, fps=fps)

    yield video_path, "🧠 Video ready. Scoring with TribeV2 (~2-4 min)...", None

    # ------------------------------------------------------------------
    # Step 2: TribeV2 brain scoring (always local)
    # ------------------------------------------------------------------
    _load_tribe()
    roi_scores: dict[str, float] = {}
    composite = 0.0
    try:
        roi_scores = _tribe_reward.score_video_file(video_path)
        weights = [CFG["rois"][k].get("weight", 1.0) for k in roi_scores]
        vals = list(roi_scores.values())
        composite = sum(w * v for w, v in zip(weights, vals)) / sum(weights)
    except Exception as e:
        logger.warning("TribeV2 scoring failed: %s", e)
        roi_scores = {"error": str(e)}

    # ------------------------------------------------------------------
    # Step 3: Brain heatmap
    # ------------------------------------------------------------------
    yield video_path, "🗺️ Rendering brain heatmap...", None
    heatmap_path = None
    try:
        from neuroadgen.visualization.brain_heatmap import generate_brain_heatmap
        vertex_preds = _tribe_reward.get_full_vertex_predictions(video_path)
        heatmap_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        heatmap_tmp.close()
        generate_brain_heatmap(
            vertex_predictions=vertex_preds,
            roi_config=CFG["rois"],
            output_png=heatmap_tmp.name,
            title=f"Brain Activation: {prompt[:60]}",
        )
        heatmap_path = heatmap_tmp.name
    except Exception as e:
        logger.warning("Heatmap failed: %s", e)

    scores_display = json.dumps(
        {"composite_score": round(composite, 4), **{k: round(v, 4) for k, v in roi_scores.items()}},
        indent=2,
    )
    yield video_path, scores_display, heatmap_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    import gradio as gr  # noqa: keep local to avoid slow top-level import

    with gr.Blocks(title="NeuroAdGen") as demo:

        gr.Markdown("# NeuroAdGen")
        gr.Markdown(
            "Generate a video and score it with **TribeV2** — Meta's brain encoding model "
            "that predicts fMRI cortical responses across ~20k brain vertices. "
            "**API mode** uses fal.ai for video generation (~30-60s). "
            "Set `FAL_KEY` env var before launching.",
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_in = gr.Textbox(
                    label="Prompt",
                    placeholder="athlete running through a neon-lit city at night, cinematic, 4K",
                    lines=3,
                )
                with gr.Row():
                    guidance = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="Guidance Scale")
                    use_api = gr.Checkbox(value=False, label="Use fal.ai API instead (needs FAL_KEY)")
                run_btn = gr.Button("Generate + Score", variant="primary")

            with gr.Column(scale=3):
                status_out = gr.Textbox(label="Status", interactive=False, lines=2)
                video_out = gr.Video(label="Generated Video")
                scores_out = gr.Code(label="TribeV2 Brain Scores", language="json")
                heatmap_out = gr.Image(label="Cortical Activation Heatmap")

        run_btn.click(
            fn=run_pipeline,
            inputs=[prompt_in, guidance, use_api],
            outputs=[video_out, status_out, heatmap_out],
        )

        gr.Markdown(
            "---\n"
            "**Ethics:** TribeV2 is trained on consented fMRI data. "
            "No real-time brain data is accessed. Do not use for covert manipulation."
        )

    return demo


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    import gradio as gr
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=".container { max-width: 1100px; margin: auto; } #title { text-align: center; margin-bottom: 8px; }",
    )

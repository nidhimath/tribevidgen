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


def _load_models():
    global _generator, _tribe_reward
    with _models_lock:
        if _generator is not None:
            return

        from neuroadgen.models.video_gen import VideoGenerator
        from neuroadgen.models.tribe_reward import TribeReward

        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32  # MPS/CPU safe

        logger.info("Loading VideoGenerator on %s...", device)
        _generator = VideoGenerator(
            model_id=CFG["video_model"]["name"],
            dtype=dtype,
            device=device,
            cache_dir=CFG["paths"]["cache_dir"],
            enable_gradient_checkpointing=False,
            truncated_bptt_k=CFG["reward_opt"]["truncated_bptt_k"],
        )

        logger.info("Loading TribeReward...")
        _tribe_reward = TribeReward(
            roi_config=CFG["rois"],
            differentiability_strategy="reinforce",  # black-box scoring for UI
            cache_folder=CFG["tribe"]["cache_folder"],
            hemodynamic_lag_sec=CFG["tribe"]["hemodynamic_lag_sec"],
            prediction_end_sec=CFG["tribe"]["prediction_end_sec"],
            fps=CFG["video_model"]["fps"],
            device=device,
        )
        logger.info("Models ready.")


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

def run_pipeline(prompt: str, n_opt_steps: int, guidance_scale: float):
    """
    Generate a video from prompt, score it with TribeV2, return results.
    Runs reward optimisation for n_opt_steps if > 0.
    """
    if not prompt.strip():
        return None, "{}", None

    yield None, "⏳ Loading models...", None

    _load_models()

    device = next(_generator.pipe.transformer.parameters()).device
    fps = CFG["video_model"]["fps"]

    # ------------------------------------------------------------------
    # Generate initial video
    # ------------------------------------------------------------------
    yield None, "🎬 Generating video...", None

    with torch.no_grad():
        video, latents = _generator.generate(
            prompt=prompt,
            num_inference_steps=CFG["video_model"]["inference_steps"],
            guidance_scale=guidance_scale,
            height=CFG["video_model"]["resolution"],
            width=int(CFG["video_model"]["resolution"] * 16 / 9),
            num_frames=CFG["video_model"]["duration_frames"],
        )

    # ------------------------------------------------------------------
    # Reward optimisation loop (optional)
    # ------------------------------------------------------------------
    if n_opt_steps > 0:
        from neuroadgen.training.reward_optimize import RewardOptimizer

        # Minimal ref generator (same weights, frozen)
        ref_gen = _generator  # KL against self — conservative

        yield None, f"🧠 Running {n_opt_steps} reward optimisation steps...", None

        opt_cfg = {**CFG, "reward_opt": {**CFG["reward_opt"], "n_steps": n_opt_steps, "batch_size": 1}}
        optimizer = RewardOptimizer(_generator, _tribe_reward, ref_gen, opt_cfg, device=str(device))
        optimizer.optimise(prompts=[prompt])

        with torch.no_grad():
            video, _ = _generator.generate(
                prompt=prompt,
                num_inference_steps=CFG["video_model"]["inference_steps"],
                guidance_scale=guidance_scale,
                height=CFG["video_model"]["resolution"],
                width=int(CFG["video_model"]["resolution"] * 16 / 9),
                num_frames=CFG["video_model"]["duration_frames"],
            )

    # ------------------------------------------------------------------
    # Save video
    # ------------------------------------------------------------------
    yield None, "💾 Saving video...", None
    video_path = _save_video(video, fps=fps)

    # ------------------------------------------------------------------
    # TribeV2 brain scoring
    # ------------------------------------------------------------------
    yield None, "🧠 Scoring with TribeV2...", None

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
    # Brain heatmap
    # ------------------------------------------------------------------
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

    scores_display = json.dumps({"composite_score": round(composite, 4), **{k: round(v, 4) for k, v in roi_scores.items()}}, indent=2)
    yield video_path, scores_display, heatmap_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    import gradio as gr  # noqa: keep local to avoid slow top-level import

    with gr.Blocks(title="NeuroAdGen") as demo:

        gr.Markdown("# 🧠 NeuroAdGen", elem_id="title")
        gr.Markdown(
            "Enter a video prompt. NeuroAdGen generates the video and scores it with "
            "**TribeV2** — a brain encoding model that predicts fMRI cortical responses "
            "across ~20k brain vertices. Higher scores = stronger predicted neural engagement.",
            elem_id="title",
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_in = gr.Textbox(
                    label="Prompt",
                    placeholder="athlete running through a neon-lit city at night, cinematic, 4K",
                    lines=3,
                )
                with gr.Row():
                    guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
                    opt_steps = gr.Slider(0, 30, value=0, step=5, label="Reward Optimisation Steps (0 = skip)")
                run_btn = gr.Button("Generate + Score", variant="primary", scale=1)

            with gr.Column(scale=3):
                status_out = gr.Textbox(label="Status", interactive=False, lines=1)
                video_out = gr.Video(label="Generated Video")
                scores_out = gr.Code(label="TribeV2 Brain Scores", language="json")
                heatmap_out = gr.Image(label="Cortical Activation Heatmap")

        run_btn.click(
            fn=run_pipeline,
            inputs=[prompt_in, opt_steps, guidance],
            outputs=[video_out, scores_out, heatmap_out],
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

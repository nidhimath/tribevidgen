"""
End-to-end NeuroAdGen inference pipeline.

Usage (Python):
    from neuroadgen.inference.generate import generate_ad
    result = generate_ad(brief)

Usage (CLI):
    python inference/generate.py \
        --brand "Nike" \
        --product "Air Max 2026" \
        --target_emotion "excitement and aspiration" \
        --scene "athlete running through neon-lit city at night" \
        --target_rois visual_engagement emotional_valence memory_encoding \
        --output_dir ./outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Prompt expansion
# ---------------------------------------------------------------------------

def expand_brief_to_prompt(brief: dict, llm_model_id: Optional[str] = None) -> str:
    """
    Convert an ad brief dict into a detailed video generation prompt.
    Uses a local LLM if available; falls back to a template.
    """
    template_prompt = (
        f"A high-quality advertising video for {brief['brand']} {brief['product']}. "
        f"Scene: {brief['scene_description']}. "
        f"Emotional tone: {brief['target_emotion']}. "
        f"Cinematic, professional production quality, 4K, vibrant colors, "
        f"dynamic camera movement, compelling narrative arc."
    )

    if llm_model_id is None:
        return template_prompt

    try:
        from transformers import pipeline
        generator = pipeline("text-generation", model=llm_model_id, device_map="auto", max_new_tokens=150)
        system_prompt = (
            "You are a creative director. Expand this advertising brief into a detailed, "
            "cinematic video prompt for a text-to-video AI model. Be specific about visuals, "
            "camera angles, lighting, and mood. Keep it under 150 words.\n\n"
            f"Brand: {brief['brand']}\n"
            f"Product: {brief['product']}\n"
            f"Emotion: {brief['target_emotion']}\n"
            f"Scene: {brief['scene_description']}\n\n"
            "Expanded prompt:"
        )
        result = generator(system_prompt, do_sample=True, temperature=0.7)
        expanded = result[0]["generated_text"].split("Expanded prompt:")[-1].strip()
        return expanded if len(expanded) > 20 else template_prompt
    except Exception as exc:
        logger.warning("LLM prompt expansion failed: %s — using template.", exc)
        return template_prompt


# ---------------------------------------------------------------------------
# Video export helper
# ---------------------------------------------------------------------------

def save_video(video: torch.Tensor, path: str, fps: int = 8) -> str:
    """Save (T, C, H, W) float tensor in [0,1] to an MP4 file."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("opencv-python required: pip install opencv-python")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    T, C, H, W = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    frames_np = (video.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
    for frame in frames_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return str(path)


# ---------------------------------------------------------------------------
# Main generate_ad function
# ---------------------------------------------------------------------------

def generate_ad(brief: dict, config_path: str | None = None) -> dict:
    """
    Full end-to-end NeuroAdGen pipeline.

    Parameters
    ----------
    brief : dict with keys:
        brand, product, target_emotion, scene_description,
        duration_seconds, target_brain_regions, roi_weights,
        reference_image (optional)
    config_path : path to YAML config (uses default if None)

    Returns
    -------
    dict with keys:
        video_path, brain_heatmap_path, roi_scores,
        composite_score, optimization_trajectory
    """
    cfg_path = config_path or str(DEFAULT_CONFIG_PATH)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(brief.get("output_dir", cfg["paths"]["outputs_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{brief['brand'].replace(' ', '_')}_{int(time.time())}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if cfg["video_model"]["dtype"] == "bf16" else torch.float16

    logger.info("=== NeuroAdGen Pipeline Start: %s ===", run_id)

    # ------------------------------------------------------------------
    # Step 1: Expand brief into detailed prompt
    # ------------------------------------------------------------------
    logger.info("Step 1: Expanding ad brief into video prompt...")
    prompt = expand_brief_to_prompt(brief)
    logger.info("Prompt: %s", prompt)

    # ------------------------------------------------------------------
    # Step 2: Load models
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading video generation model...")
    from neuroadgen.models.video_gen import VideoGenerator
    from neuroadgen.models.tribe_reward import TribeReward
    from neuroadgen.models.lora_adapter import load_lora_checkpoint
    from neuroadgen.training.reward_optimize import RewardOptimizer
    from neuroadgen.visualization.brain_heatmap import generate_brain_heatmap

    generator = VideoGenerator(
        model_id=cfg["video_model"]["name"],
        dtype=dtype,
        device=device,
        cache_dir=cfg["paths"]["cache_dir"],
        enable_gradient_checkpointing=cfg["hardware"]["gradient_checkpointing"],
        truncated_bptt_k=cfg["reward_opt"]["truncated_bptt_k"],
    )

    # Load LoRA checkpoint if available
    lora_ckpt_dir = cfg["paths"]["checkpoints_dir"]
    latest_ckpt = _find_latest_checkpoint(lora_ckpt_dir)
    if latest_ckpt:
        logger.info("Loading LoRA checkpoint: %s", latest_ckpt)
        generator.pipe.transformer = load_lora_checkpoint(
            generator.pipe.transformer, latest_ckpt, device=device
        )
    else:
        logger.info("No LoRA checkpoint found — using base model weights.")

    # Reference model for KL penalty (frozen copy)
    ref_generator = VideoGenerator(
        model_id=cfg["video_model"]["name"],
        dtype=dtype,
        device=device,
        cache_dir=cfg["paths"]["cache_dir"],
    )
    if latest_ckpt:
        ref_generator.pipe.transformer = load_lora_checkpoint(
            ref_generator.pipe.transformer, latest_ckpt, device=device
        )

    # Build ROI config filtered to requested target regions
    target_rois = brief.get("target_brain_regions", list(cfg["rois"].keys()))
    roi_weights_override = brief.get("roi_weights", {})
    roi_config = {
        k: {**v, "weight": roi_weights_override.get(k, v.get("weight", 1.0))}
        for k, v in cfg["rois"].items()
        if k in target_rois
    }

    logger.info("Loading TribeV2 reward model...")
    tribe_reward = TribeReward(
        roi_config=roi_config,
        differentiability_strategy=cfg["reward_opt"]["differentiability_strategy"],
        cache_folder=cfg["tribe"]["cache_folder"],
        hemodynamic_lag_sec=cfg["tribe"]["hemodynamic_lag_sec"],
        prediction_end_sec=cfg["tribe"]["prediction_end_sec"],
        fps=cfg["video_model"]["fps"],
        device=device,
        vjepa2_model_id=cfg["reward_opt"].get("vjepa2_model_id"),
    )

    # ------------------------------------------------------------------
    # Step 3: Generate initial video
    # ------------------------------------------------------------------
    logger.info("Step 3: Generating initial video...")
    with torch.no_grad():
        video_init, _ = generator.generate(
            prompt=prompt,
            num_inference_steps=cfg["video_model"]["inference_steps"],
            guidance_scale=cfg["video_model"]["guidance_scale"],
            height=cfg["video_model"]["resolution"],
            width=cfg["video_model"]["resolution"] * 16 // 9,
            num_frames=cfg["video_model"]["duration_frames"],
        )

    init_video_path = str(output_dir / f"{run_id}_initial.mp4")
    save_video(video_init, init_video_path, fps=cfg["video_model"]["fps"])
    logger.info("Initial video saved: %s", init_video_path)

    # ------------------------------------------------------------------
    # Step 4: Reward optimisation loop
    # ------------------------------------------------------------------
    logger.info("Step 4: Running reward optimisation (%d steps)...", cfg["reward_opt"]["n_steps"])
    optimizer = RewardOptimizer(
        generator=generator,
        tribe_reward=tribe_reward,
        ref_generator=ref_generator,
        cfg=cfg,
        device=device,
    )

    trajectory = optimizer.optimise(
        prompts=[prompt],
        roi_weights=roi_weights_override or None,
    )

    # ------------------------------------------------------------------
    # Step 5: Generate final optimised video
    # ------------------------------------------------------------------
    logger.info("Step 5: Generating final optimised video...")
    with torch.no_grad():
        video_final, _ = generator.generate(
            prompt=prompt,
            num_inference_steps=cfg["video_model"]["inference_steps"],
            guidance_scale=cfg["video_model"]["guidance_scale"],
            height=cfg["video_model"]["resolution"],
            width=cfg["video_model"]["resolution"] * 16 // 9,
            num_frames=cfg["video_model"]["duration_frames"],
        )

    final_video_path = str(output_dir / f"{run_id}_optimized.mp4")
    save_video(video_final, final_video_path, fps=cfg["video_model"]["fps"])
    logger.info("Optimised video saved: %s", final_video_path)

    # ------------------------------------------------------------------
    # Step 6: Final TribeV2 scoring (black-box, full prediction)
    # ------------------------------------------------------------------
    logger.info("Step 6: Running final TribeV2 brain scoring...")
    roi_scores: dict[str, float] = {}
    vertex_predictions = None
    try:
        roi_scores = tribe_reward.score_video_file(final_video_path)
        vertex_predictions = tribe_reward.get_full_vertex_predictions(final_video_path)
    except Exception as exc:
        logger.warning("TribeV2 final scoring failed: %s — using proxy scores from trajectory.", exc)
        if trajectory:
            roi_scores = trajectory[-1]["roi_scores"]

    weights = torch.tensor(
        [roi_weights_override.get(k, roi_config[k].get("weight", 1.0)) for k in roi_scores],
        dtype=torch.float32,
    )
    scores_t = torch.tensor(list(roi_scores.values()), dtype=torch.float32)
    composite_score = float((weights * scores_t).sum() / weights.sum())

    # ------------------------------------------------------------------
    # Step 7: Brain heatmap
    # ------------------------------------------------------------------
    logger.info("Step 7: Generating brain heatmap...")
    heatmap_path = str(output_dir / f"{run_id}_brain_heatmap.png")
    interactive_path = str(output_dir / f"{run_id}_brain_interactive.html")
    try:
        generate_brain_heatmap(
            vertex_predictions=vertex_predictions,
            roi_config=roi_config,
            output_png=heatmap_path,
            output_html=interactive_path,
        )
    except Exception as exc:
        logger.warning("Brain heatmap generation failed: %s", exc)
        heatmap_path = None

    # ------------------------------------------------------------------
    # Step 8: Save results
    # ------------------------------------------------------------------
    result = {
        "video_path": final_video_path,
        "initial_video_path": init_video_path,
        "brain_heatmap_path": heatmap_path,
        "brain_interactive_path": interactive_path if heatmap_path else None,
        "roi_scores": roi_scores,
        "composite_score": composite_score,
        "optimization_trajectory": trajectory,
        "prompt": prompt,
        "run_id": run_id,
    }

    results_path = str(output_dir / f"{run_id}_results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=== Pipeline Complete ===")
    logger.info("Video:          %s", final_video_path)
    logger.info("Heatmap:        %s", heatmap_path)
    logger.info("Composite score: %.4f", composite_score)
    logger.info("ROI scores:     %s", roi_scores)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return path to latest lora_step_* checkpoint, or None."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None
    ckpts = sorted(ckpt_path.glob("lora_step_*"))
    return str(ckpts[-1]) if ckpts else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroAdGen — brain-optimised ad video generation.")
    p.add_argument("--brand", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--target_emotion", required=True)
    p.add_argument("--scene", required=True, dest="scene_description")
    p.add_argument("--duration", type=float, default=15.0, dest="duration_seconds")
    p.add_argument("--target_rois", nargs="+",
                   default=["visual_engagement", "emotional_valence", "memory_encoding"])
    p.add_argument("--roi_weights", nargs="+", type=float,
                   help="Weights matching --target_rois order (e.g. 0.4 0.4 0.2)")
    p.add_argument("--reference_image", default=None)
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--config", default=None)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    args = parse_args()

    roi_w: dict[str, float] = {}
    if args.roi_weights:
        if len(args.roi_weights) != len(args.target_rois):
            raise ValueError("--roi_weights must have same length as --target_rois")
        roi_w = dict(zip(args.target_rois, args.roi_weights))

    brief = {
        "brand": args.brand,
        "product": args.product,
        "target_emotion": args.target_emotion,
        "scene_description": args.scene_description,
        "duration_seconds": args.duration_seconds,
        "target_brain_regions": args.target_rois,
        "roi_weights": roi_w or {},
        "reference_image": args.reference_image,
        "output_dir": args.output_dir,
    }

    result = generate_ad(brief, config_path=args.config)

    print("\n=== Results ===")
    print(f"Video:            {result['video_path']}")
    print(f"Brain heatmap:    {result['brain_heatmap_path']}")
    print(f"Composite score:  {result['composite_score']:.4f}")
    print(f"ROI scores:       {result['roi_scores']}")

"""
Stage 2: VADER-style reward gradient optimisation.

Runs AFTER Stage 1 LoRA fine-tuning. Iteratively adjusts LoRA weights
to maximise brain ROI activations predicted by TribeV2.

Key design choices:
  - Truncated BPTT through the last K=5 denoising steps (VRAM budget).
  - KL divergence penalty between optimised model and Stage-1 baseline
    to prevent reward hacking.
  - Differentiability via V-JEPA2 proxy by default (Strategy A).
  - Gradient clipping at max_norm=1.0.
  - Adam optimiser at lr=1e-5.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward Optimiser
# ---------------------------------------------------------------------------

class RewardOptimizer:
    """
    VADER-style brain reward gradient optimisation loop.

    Parameters
    ----------
    generator     : VideoGenerator — fine-tuned video model with LoRA.
    tribe_reward  : TribeReward — neural reward model.
    ref_generator : VideoGenerator — frozen Stage-1 baseline for KL penalty.
    cfg           : Full config dict.
    device        : Compute device.
    """

    def __init__(
        self,
        generator,
        tribe_reward,
        ref_generator,
        cfg: dict,
        device: str | torch.device = "cuda",
    ) -> None:
        self.generator = generator
        self.tribe_reward = tribe_reward
        self.ref_generator = ref_generator
        self.cfg = cfg
        self.device = torch.device(device)

        opt_cfg = cfg["reward_opt"]
        self.n_steps: int = opt_cfg["n_steps"]
        self.batch_size: int = opt_cfg["batch_size"]
        self.lr: float = opt_cfg["learning_rate"]
        self.kl_weight: float = opt_cfg["kl_penalty_weight"]
        self.grad_clip: float = opt_cfg["grad_clip_norm"]
        self.k_bptt: int = opt_cfg["truncated_bptt_k"]

        # Only optimise LoRA parameters
        lora_params = [p for n, p in generator.pipe.transformer.named_parameters()
                       if "lora_" in n and p.requires_grad]
        if not lora_params:
            logger.warning("No LoRA parameters found — all params will be optimised.")
            lora_params = [p for p in generator.pipe.transformer.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(lora_params, lr=self.lr)
        logger.info("RewardOptimiser: %d LoRA params, lr=%.1e, K=%d, kl_w=%.3f",
                    sum(p.numel() for p in lora_params), self.lr, self.k_bptt, self.kl_weight)

        # Freeze reference model
        for p in self.ref_generator.pipe.transformer.parameters():
            p.requires_grad_(False)

        self.trajectory: list[dict] = []

    # ------------------------------------------------------------------
    # KL penalty
    # ------------------------------------------------------------------

    def _kl_penalty(self, noisy_latents: torch.Tensor, encoder_hidden_states: torch.Tensor,
                    timesteps: torch.Tensor) -> torch.Tensor:
        """
        Approximate KL divergence between optimised model and reference model
        by comparing noise prediction distributions.

        KL ≈ 0.5 * ||μ_opt - μ_ref||² / σ²
        """
        with torch.no_grad():
            ref_pred = self.ref_generator.pipe.transformer(
                noisy_latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
            ).sample

        opt_pred = self.generator.pipe.transformer(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
        ).sample

        kl = 0.5 * F.mse_loss(opt_pred, ref_pred)
        return kl

    # ------------------------------------------------------------------
    # Video-to-file helper for TribeV2 black-box path
    # ------------------------------------------------------------------

    @staticmethod
    def _save_video_temp(video: torch.Tensor, fps: int = 8) -> str:
        """Save (T, C, H, W) float tensor in [0,1] to a temp MP4 file."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ImportError("opencv-python required for video export.")

        tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmppath = tmpfile.name
        tmpfile.close()

        T, C, H, W = video.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmppath, fourcc, fps, (W, H))
        frames_np = (video.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
        for frame in frames_np:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return tmppath

    # ------------------------------------------------------------------
    # Single optimisation step
    # ------------------------------------------------------------------

    def _step(
        self,
        prompt: str,
        roi_weights: Optional[dict[str, float]] = None,
    ) -> dict:
        """Run one reward optimisation step for a single prompt."""
        video_cfg = self.cfg["video_model"]

        # Generate video (last K steps differentiable via truncated BPTT)
        video, latents = self.generator.generate(
            prompt=prompt,
            num_inference_steps=video_cfg["inference_steps"],
            guidance_scale=video_cfg["guidance_scale"],
            height=video_cfg["resolution"],
            width=video_cfg["resolution"] * 16 // 9,
            num_frames=video_cfg["duration_frames"],
        )
        # video: (T, C, H, W) in [0,1], grad attached through last K denoising steps

        # Ensure video is at least min_input_duration_sec for TribeV2
        min_frames = int(self.cfg["tribe"]["min_input_duration_sec"] * video_cfg["fps"])
        if video.shape[0] < min_frames:
            # Loop/pad the video
            reps = (min_frames // video.shape[0]) + 1
            video = video.repeat(reps, 1, 1, 1)[:min_frames]

        # Compute brain reward (differentiable via strategy A or C)
        strategy = self.cfg["reward_opt"]["differentiability_strategy"]
        video_path = None
        if strategy == "reinforce":
            video_path = self._save_video_temp(video, fps=video_cfg["fps"])

        reward, roi_scores = self.tribe_reward(
            video=video,
            video_path=video_path,
            roi_weights=roi_weights,
        )

        # KL penalty — requires one forward pass through both models
        # Use a noise sample at a random timestep
        scheduler = self.generator.pipe.scheduler
        tokenizer = self.generator.pipe.tokenizer
        text_encoder = self.generator.pipe.text_encoder

        with torch.no_grad():
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            enc_hidden = text_encoder(**text_inputs).last_hidden_state

        noisy = torch.randn_like(latents[:, :, :1])
        t_kl = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=self.device)
        kl = self._kl_penalty(noisy, enc_hidden, t_kl)

        # Total loss: maximise reward, minimise KL
        loss = -reward + self.kl_weight * kl

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.generator.pipe.transformer.parameters() if p.requires_grad],
            self.grad_clip,
        )
        self.optimizer.step()

        # Cleanup temp file if used
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)

        return {
            "loss": loss.item(),
            "reward": reward.item(),
            "kl": kl.item(),
            "roi_scores": roi_scores,
        }

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def optimise(
        self,
        prompts: list[str],
        roi_weights: Optional[dict[str, float]] = None,
        wandb_run=None,
    ) -> list[dict]:
        """
        Run the full reward optimisation loop.

        Parameters
        ----------
        prompts    : List of prompts to sample from each step
                     (batch_size prompts per step, cycling through list).
        roi_weights: Per-ROI weighting for composite reward.

        Returns
        -------
        trajectory : List of per-step metric dicts.
        """
        logger.info("Starting reward optimisation: %d steps, %d prompts.", self.n_steps, len(prompts))
        self.trajectory = []
        best_reward = -float("inf")
        best_lora_state = None

        for step in range(self.n_steps):
            step_metrics: list[dict] = []
            for i in range(self.batch_size):
                prompt = prompts[(step * self.batch_size + i) % len(prompts)]
                metrics = self._step(prompt, roi_weights)
                step_metrics.append(metrics)

            avg = {
                "step": step,
                "reward": sum(m["reward"] for m in step_metrics) / len(step_metrics),
                "loss": sum(m["loss"] for m in step_metrics) / len(step_metrics),
                "kl": sum(m["kl"] for m in step_metrics) / len(step_metrics),
                "roi_scores": step_metrics[-1]["roi_scores"],
            }
            self.trajectory.append(avg)

            logger.info(
                "Step %d/%d — reward: %.4f  loss: %.4f  kl: %.4f",
                step + 1, self.n_steps, avg["reward"], avg["loss"], avg["kl"],
            )

            if wandb_run:
                wandb_run.log({
                    "reward/composite": avg["reward"],
                    "reward/kl": avg["kl"],
                    "train/loss": avg["loss"],
                    **{f"roi/{k}": v for k, v in avg["roi_scores"].items()},
                }, step=step)

            # Track best model
            if avg["reward"] > best_reward:
                best_reward = avg["reward"]
                best_lora_state = {
                    k: v.clone()
                    for k, v in self.generator.pipe.transformer.state_dict().items()
                    if "lora_" in k
                }

        # Restore best checkpoint
        if best_lora_state is not None:
            self.generator.pipe.transformer.load_state_dict(best_lora_state, strict=False)
            logger.info("Restored best LoRA weights (reward=%.4f).", best_reward)

        return self.trajectory


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--lora_ckpt", required=True, help="Path to Stage-1 LoRA checkpoint.")
    p.add_argument("--prompts_file", required=True, help="Text file with one prompt per line.")
    p.add_argument("--output_dir", default="./outputs/reward_opt")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prompts = Path(args.prompts_file).read_text().strip().splitlines()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid circular deps
    from neuroadgen.models.video_gen import VideoGenerator
    from neuroadgen.models.tribe_reward import TribeReward
    from neuroadgen.models.lora_adapter import load_lora_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    gen = VideoGenerator(model_id=cfg["video_model"]["name"], dtype=dtype, device=device,
                         cache_dir=cfg["paths"]["cache_dir"],
                         truncated_bptt_k=cfg["reward_opt"]["truncated_bptt_k"])
    gen.pipe.transformer = load_lora_checkpoint(gen.pipe.transformer, args.lora_ckpt, device=device)

    ref_gen = VideoGenerator(model_id=cfg["video_model"]["name"], dtype=dtype, device=device,
                              cache_dir=cfg["paths"]["cache_dir"])
    ref_gen.pipe.transformer = load_lora_checkpoint(ref_gen.pipe.transformer, args.lora_ckpt, device=device)

    reward_model = TribeReward(
        roi_config=cfg["rois"],
        differentiability_strategy=cfg["reward_opt"]["differentiability_strategy"],
        cache_folder=cfg["tribe"]["cache_folder"],
        device=device,
    )

    opt = RewardOptimizer(gen, reward_model, ref_gen, cfg, device=device)
    traj = opt.optimise(prompts)

    import json
    with open(Path(args.output_dir) / "trajectory.json", "w") as f:
        json.dump(traj, f, indent=2)
    logger.info("Optimisation complete. Trajectory saved.")

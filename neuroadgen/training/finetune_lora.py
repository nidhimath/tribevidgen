"""
Stage 1: Supervised LoRA fine-tuning on ad video dataset.

Usage (single GPU):
    python training/finetune_lora.py --config configs/default.yaml

Usage (multi-GPU with accelerate + DeepSpeed ZeRO-2):
    accelerate launch --config_file configs/accelerate_zero2.yaml \
        training/finetune_lora.py --config configs/default.yaml

Dataset format: a directory of (prompt.txt, video.mp4) pairs, or a metadata
JSON file with {"prompt": "...", "video_path": "..."} records.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AdVideoDataset(Dataset):
    """
    Loads (text_prompt, video_clip) pairs for supervised LoRA fine-tuning.

    Expects either:
      - A directory of *.mp4 files each accompanied by a same-stem *.txt caption.
      - A metadata.json with list of {"prompt": "...", "video_path": "..."} records.
    """

    def __init__(
        self,
        data_dir: str,
        fps: int = 8,
        n_frames: int = 120,
        resolution: int = 480,
        augment_captions: bool = False,
        llm_caption_model: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.fps = fps
        self.n_frames = n_frames
        self.resolution = resolution
        self.augment_captions = augment_captions

        self.pairs = self._build_pairs()
        logger.info("AdVideoDataset: %d samples from %s", len(self.pairs), data_dir)

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution * 16 // 9)),
            transforms.CenterCrop((resolution, resolution * 16 // 9)),
        ])

        self._llm = None
        if augment_captions and llm_caption_model:
            self._llm = self._load_llm(llm_caption_model)

    def _build_pairs(self) -> list[dict]:
        pairs = []
        meta_file = self.data_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                pairs = json.load(f)
        else:
            for video_path in sorted(self.data_dir.glob("**/*.mp4")):
                caption_path = video_path.with_suffix(".txt")
                if caption_path.exists():
                    pairs.append({
                        "prompt": caption_path.read_text().strip(),
                        "video_path": str(video_path),
                    })
                else:
                    logger.warning("No caption for %s — skipping.", video_path)
        return pairs

    def _load_llm(self, model_id: str):
        from transformers import pipeline
        return pipeline("text-generation", model=model_id, device_map="auto")

    def _augment_caption(self, brief: str) -> str:
        """Expand a short ad brief into a detailed scene description via LLM."""
        if self._llm is None:
            return brief
        prompt = (
            f"Expand this advertising brief into a detailed, cinematic video scene description "
            f"of 2-3 sentences. Brief: '{brief}'\nExpanded:"
        )
        result = self._llm(prompt, max_new_tokens=80, do_sample=True, temperature=0.7)
        return result[0]["generated_text"].split("Expanded:")[-1].strip()

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load video, sample at self.fps, return (T, C, H, W) in [-1, 1]."""
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")

        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
        stride = max(1, round(src_fps / self.fps))

        frames = []
        idx = 0
        while len(frames) < self.n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(t)
            idx += 1
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Could not read frames from {video_path}")

        # Pad or truncate to n_frames
        while len(frames) < self.n_frames:
            frames.append(frames[-1])
        frames = frames[:self.n_frames]

        video = torch.stack(frames)  # (T, C, H, W)
        # Resize
        T, C, H, W = video.shape
        video = F.interpolate(video, size=(self.resolution, self.resolution * 16 // 9), mode="bilinear", align_corners=False)
        # Normalize to [-1, 1]
        video = video * 2 - 1
        return video  # (T, C, H, W)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        item = self.pairs[idx]
        prompt = item["prompt"]
        if self.augment_captions:
            prompt = self._augment_caption(prompt)
        video = self._load_video_frames(item["video_path"])
        return {"prompt": prompt, "video": video}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    from accelerate import Accelerator
    from transformers import get_cosine_schedule_with_warmup
    import wandb

    from neuroadgen.models.video_gen import VideoGenerator
    from neuroadgen.models.lora_adapter import inject_lora, freeze_base_weights, save_lora_checkpoint

    # Accelerator handles mixed precision + DeepSpeed
    accelerator = Accelerator(
        mixed_precision=cfg["hardware"]["mixed_precision"],
        gradient_accumulation_steps=cfg["finetune"]["gradient_accumulation"],
        log_with="wandb",
    )

    # Init wandb
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg["wandb"]["project"],
            config=cfg,
        )

    # Load video model
    generator = VideoGenerator(
        model_id=cfg["video_model"]["name"],
        dtype=torch.bfloat16 if cfg["video_model"]["dtype"] == "bf16" else torch.float16,
        device=accelerator.device,
        cache_dir=cfg["paths"]["cache_dir"],
        enable_gradient_checkpointing=cfg["hardware"]["gradient_checkpointing"],
    )
    transformer = generator.pipe.transformer

    # Inject LoRA
    lora_cfg = cfg["lora"]
    transformer = inject_lora(
        transformer,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
    )
    freeze_base_weights(transformer)

    # Dataset + DataLoader
    dataset = AdVideoDataset(
        data_dir=cfg.get("data_dir", "data/ad_dataset"),
        fps=cfg["video_model"]["fps"],
        n_frames=cfg["video_model"]["duration_frames"],
        resolution=cfg["video_model"]["resolution"],
        augment_captions=cfg.get("augment_captions", False),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["finetune"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=cfg["finetune"]["learning_rate"],
        weight_decay=cfg["finetune"]["weight_decay"],
    )
    total_steps = cfg["finetune"]["train_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["finetune"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    transformer, optimizer, dataloader, scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, scheduler
    )

    noise_scheduler = generator.pipe.scheduler
    vae = generator.pipe.vae
    text_encoder = generator.pipe.text_encoder
    tokenizer = generator.pipe.tokenizer

    global_step = 0
    logger.info("Starting LoRA fine-tuning for %d steps.", total_steps)

    while global_step < total_steps:
        for batch in dataloader:
            if global_step >= total_steps:
                break

            prompts = batch["prompt"]
            videos = batch["video"].to(accelerator.device, dtype=torch.bfloat16)  # (B, T, C, H, W)

            # Encode video to latents via VAE
            B, T, C, H, W = videos.shape
            with torch.no_grad():
                # Rearrange to (B, C, T, H, W) for VAE
                video_input = videos.permute(0, 2, 1, 3, 4)
                latents = vae.encode(video_input).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Encode text
            with torch.no_grad():
                text_inputs = tokenizer(
                    list(prompts),
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)
                encoder_hidden_states = text_encoder(**text_inputs).last_hidden_state

            # Sample noise and timestep
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=latents.device
            ).long()

            # Add noise (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            with accelerator.accumulate(transformer):
                noise_pred = transformer(
                    noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                ).sample

                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(),
                        cfg["finetune"]["max_grad_norm"],
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % 10 == 0 and accelerator.is_main_process:
                    logger.info("Step %d / %d — loss: %.4f", global_step, total_steps, loss.item())
                    accelerator.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

                if global_step % cfg["finetune"]["save_every"] == 0 and accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(transformer)
                    save_lora_checkpoint(
                        unwrapped,
                        checkpoint_dir=cfg["paths"]["checkpoints_dir"],
                        step=global_step,
                        metadata={"loss": loss.item()},
                    )

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        save_lora_checkpoint(
            unwrapped,
            checkpoint_dir=cfg["paths"]["checkpoints_dir"],
            step=global_step,
            metadata={"final": True},
        )
        logger.info("LoRA fine-tuning complete. Final checkpoint saved.")

    accelerator.end_training()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune video model on ad dataset.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_dir", default=None, help="Override data directory from config.")
    p.add_argument("--train_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.train_steps:
        cfg["finetune"]["train_steps"] = args.train_steps
    if args.batch_size:
        cfg["finetune"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["finetune"]["learning_rate"] = args.lr

    train(cfg)

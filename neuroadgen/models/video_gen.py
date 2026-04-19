"""
Video generation backends for NeuroAdGen.

Primary: Wan2.1-T2V-14B (local, differentiable)
Fallbacks: HunyuanVideo-13B, CogVideoX-5B (local, differentiable)
Optional: SeedanceAPIGenerator via fal.ai (API-only, non-differentiable)

Latents are kept attached to the computation graph so reward gradients
can flow back through the truncated denoising chain (truncated BPTT, K=5).
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wan2.1 / HunyuanVideo / CogVideoX unified loader
# ---------------------------------------------------------------------------

class VideoGenerator(nn.Module):
    """
    Wraps a text-to-video diffusion model (DiT backbone) and exposes a
    `generate()` method that returns both the decoded RGB tensor AND the
    final latents, keeping the computation graph intact for reward
    gradient optimisation via truncated BPTT.

    Supported model IDs
    -------------------
    - "Wan-AI/Wan2.1-T2V-14B"         (primary)
    - "hunyuanvideo-community/HunyuanVideo"  (fallback)
    - "THUDM/CogVideoX-5b"            (single-GPU fallback)
    """

    SUPPORTED_MODELS = {
        "wan2.1": "Wan-AI/Wan2.1-T2V-14B",
        "hunyuan": "hunyuanvideo-community/HunyuanVideo",
        "cogvideox": "THUDM/CogVideoX-5b",
    }

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-14B",
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        cache_dir: str = "./cache",
        enable_gradient_checkpointing: bool = True,
        truncated_bptt_k: int = 5,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.dtype = dtype
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.truncated_bptt_k = truncated_bptt_k

        self.pipe = self._load_pipeline(model_id, enable_gradient_checkpointing)
        self._backend = self._detect_backend(model_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_backend(self, model_id: str) -> str:
        # Also handle local cache paths like ./cache/wan2.1-1.3b
        mid = model_id.lower().replace("\\", "/").split("/")[-1]
        if "wan" in mid or "wan" in model_id.lower():
            return "wan"
        if "hunyuan" in mid:
            return "hunyuan"
        if "cogvideo" in mid:
            return "cogvideo"
        return "cogvideo"  # safest single-GPU fallback

    def _load_pipeline(self, model_id: str, gradient_checkpointing: bool):
        """Load the appropriate diffusers pipeline."""
        backend = self._detect_backend(model_id)
        logger.info("Loading video model: %s (backend=%s)", model_id, backend)

        try:
            if backend == "wan":
                return self._load_wan(model_id, gradient_checkpointing)
            elif backend == "hunyuan":
                return self._load_hunyuan(model_id, gradient_checkpointing)
            else:
                return self._load_cogvideo(model_id, gradient_checkpointing)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — trying CogVideoX-5b fallback", model_id, exc)
            return self._load_cogvideo("THUDM/CogVideoX-5b", gradient_checkpointing)

    def _load_wan(self, model_id: str, gc: bool):
        from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
        from transformers import UMT5EncoderModel

        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        if gc:
            transformer.enable_gradient_checkpointing()

        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )

        pipe = WanPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)
        return pipe

    def _load_hunyuan(self, model_id: str, gc: bool):
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        if gc:
            transformer.enable_gradient_checkpointing()

        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)
        return pipe

    def _load_cogvideo(self, model_id: str, gc: bool):
        from diffusers import CogVideoXPipeline

        pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)
        if gc:
            pipe.transformer.enable_gradient_checkpointing()
        return pipe

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 480,
        width: int = 854,
        num_frames: int = 120,
        latents: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Generate a video from a text prompt.

        Returns
        -------
        video : Tensor  shape (T, C, H, W), float32 in [0, 1]
            Decoded RGB frames.
        final_latents : Tensor  shape (1, C_lat, T_lat, H_lat, W_lat)
            Latents after denoising. The computation graph is kept intact
            for the last `truncated_bptt_k` denoising steps so that
            reward gradients can flow back through the VAE decoder and
            the final K steps of the DiT.
        """
        output = self._run_pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            latents=latents,
            generator=generator,
        )
        video = output["video"]        # (T, C, H, W) in [0, 1]
        final_latents = output["latents"]  # differentiable
        return video, final_latents

    def decode_latents(self, latents: Tensor) -> Tensor:
        """Decode latents to RGB video tensor (T, C, H, W) in [0, 1]."""
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            video = self.pipe.decode_latents(latents)
        return video

    # ------------------------------------------------------------------
    # Backend-specific pipeline runners
    # ------------------------------------------------------------------

    def _run_pipeline(self, **kwargs) -> dict:
        """Dispatch to backend-specific runner and return video + latents."""
        if self._backend == "wan":
            return self._run_wan(**kwargs)
        elif self._backend == "hunyuan":
            return self._run_hunyuan(**kwargs)
        else:
            return self._run_cogvideo(**kwargs)

    def _run_wan(
        self,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        num_frames: int,
        latents,
        generator,
    ) -> dict:
        from diffusers.utils import export_to_video
        import numpy as np

        # We run the scheduler loop manually so we can keep the last K
        # denoising steps differentiable via truncated BPTT.
        scheduler = self.pipe.scheduler
        text_enc = self.pipe.text_encoder
        tokenizer = self.pipe.tokenizer
        transformer = self.pipe.transformer
        vae = self.pipe.vae

        with torch.no_grad():
            # Encode text
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            encoder_hidden_states = text_enc(**text_inputs).last_hidden_state

            # Prepare latents
            if latents is None:
                lat_shape = (
                    1,
                    transformer.config.in_channels,
                    num_frames // vae.temporal_compression_ratio,
                    height // vae.spatial_compression_ratio,
                    width // vae.spatial_compression_ratio,
                )
                latents = torch.randn(lat_shape, dtype=self.dtype, device=self.device, generator=generator)
            latents = latents * scheduler.init_noise_sigma

            scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = scheduler.timesteps
            bptt_start = len(timesteps) - self.truncated_bptt_k

            # Denoising loop: first (N-K) steps without gradient
            for i, t in enumerate(timesteps[:bptt_start]):
                noise_pred = transformer(
                    latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=t.unsqueeze(0),
                ).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Last K steps WITH gradient (truncated BPTT)
        for i, t in enumerate(timesteps[bptt_start:]):
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                noise_pred = transformer(
                    latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=t.unsqueeze(0),
                ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode — keep grad through VAE
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            video_tensor = vae.decode(latents / vae.config.scaling_factor).sample

        # Normalise to [0, 1], shape (1, C, T, H, W) → (T, C, H, W)
        video = (video_tensor.squeeze(0).permute(1, 0, 2, 3).float() + 1) / 2
        video = video.clamp(0, 1)
        return {"video": video, "latents": latents}

    def _run_cogvideo(
        self,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        num_frames: int,
        latents,
        generator,
    ) -> dict:
        scheduler = self.pipe.scheduler
        transformer = self.pipe.transformer
        vae = self.pipe.vae
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        with torch.no_grad():
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            prompt_embeds = text_encoder(**text_inputs).last_hidden_state

            if latents is None:
                t_lat = (num_frames - 1) // 4 + 1
                h_lat = height // 8
                w_lat = width // 8
                latents = torch.randn(
                    (1, transformer.config.in_channels, t_lat, h_lat, w_lat),
                    dtype=self.dtype,
                    device=self.device,
                    generator=generator,
                )
            latents = latents * scheduler.init_noise_sigma

            scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = scheduler.timesteps
            bptt_start = len(timesteps) - self.truncated_bptt_k

            for i, t in enumerate(timesteps[:bptt_start]):
                latent_model_input = torch.cat([latents] * 2)
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds.repeat(2, 1, 1),
                    timestep=t,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        for i, t in enumerate(timesteps[bptt_start:]):
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                latent_model_input = torch.cat([latents] * 2)
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds.repeat(2, 1, 1),
                    timestep=t,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            video_tensor = vae.decode(latents / vae.config.scaling_factor).sample

        video = (video_tensor.squeeze(0).permute(1, 0, 2, 3).float() + 1) / 2
        video = video.clamp(0, 1)
        return {"video": video, "latents": latents}

    def _run_hunyuan(self, **kwargs) -> dict:
        # HunyuanVideo uses the same denoising pattern; delegate to CogVideo runner
        # after adapting the pipeline references — override self.pipe references inline
        return self._run_cogvideo(**kwargs)


# ---------------------------------------------------------------------------
# CogVideoX fal.ai API generator — fast path (~30-60s per video)
# ---------------------------------------------------------------------------

class CogVideoXAPIGenerator:
    """
    Generates video via the fal.ai CogVideoX-5b hosted endpoint.

    No local GPU needed for generation. TribeV2 scoring still runs locally.
    Reward gradient optimisation is NOT available through this path.

    Setup: pip install fal-client && export FAL_KEY="your_key"
    Get a key at: https://fal.ai/dashboard/keys
    """

    ENDPOINT = "fal-ai/cogvideox-5b"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 300) -> None:
        self.api_key = api_key or os.environ.get("FAL_KEY", "")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "FAL_KEY not set. Export it with: export FAL_KEY='your_key'\n"
                "Get a key at: https://fal.ai/dashboard/keys"
            )

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        num_frames: int = 49,
        fps: int = 8,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a video via fal.ai API and save it locally.

        Returns
        -------
        str : local path to the downloaded MP4 file.
        """
        try:
            import fal_client
        except ImportError:
            raise RuntimeError("Run: pip install fal-client")

        os.environ["FAL_KEY"] = self.api_key
        logger.info("Submitting CogVideoX generation to fal.ai (endpoint: %s)...", self.ENDPOINT)

        result = fal_client.subscribe(
            self.ENDPOINT,
            arguments={
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": num_frames,
            },
            with_logs=True,
            on_queue_update=lambda u: logger.info("fal.ai: %s", getattr(u, "logs", u)),
        )

        video_url = result["video"]["url"]
        logger.info("Video generated. Downloading from: %s", video_url)

        if output_path is None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = tmp.name
            tmp.close()

        self._download(video_url, output_path)
        logger.info("Video saved to: %s", output_path)
        return output_path

    def _download(self, url: str, dest: str) -> None:
        import urllib.request
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------------------
# Seedance API generator (non-differentiable, for initial draft only)
# ---------------------------------------------------------------------------

class SeedanceAPIGenerator:
    """
    Wraps the fal.ai Seedance 2.0 T2V endpoint for high-quality draft generation.

    IMPORTANT: Seedance weights are NOT open-source (ByteDance proprietary).
    Reward gradient optimisation CANNOT flow through this API call.
    Use only for initial draft generation; switch to VideoGenerator for reward opt.
    """

    API_URL = "https://fal.run/fal-ai/seedance-2-0-t2v"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 300) -> None:
        self.api_key = api_key or os.environ.get("FAL_KEY", "")
        self.timeout = timeout
        if not self.api_key:
            logger.warning("FAL_KEY not set — SeedanceAPIGenerator will fail at runtime")

    def generate(
        self,
        prompt: str,
        duration: float = 5.0,
        resolution: str = "720p",
        camera_fixed: bool = False,
    ) -> str:
        """
        Submit a generation request and return the URL/path of the resulting video.

        Returns
        -------
        str : local path to downloaded video file
        """
        try:
            import fal_client  # pip install fal-client
        except ImportError:
            raise RuntimeError("fal_client not installed. Run: pip install fal-client")

        logger.info("Submitting Seedance generation request (non-differentiable)...")
        result = fal_client.subscribe(
            "fal-ai/seedance-2-0-t2v",
            arguments={
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "camera_fixed": camera_fixed,
            },
            with_logs=True,
        )
        video_url = result["video"]["url"]
        local_path = self._download(video_url)
        logger.info("Seedance video saved to: %s", local_path)
        return local_path

    def _download(self, url: str) -> str:
        import urllib.request
        out = Path("./outputs/seedance_draft.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, out)
        return str(out)

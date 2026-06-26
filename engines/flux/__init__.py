"""
engines/flux/__init__.py

Production-ready EnginePlugin for FLUX models.
- Dynamically loads a FLUX model from a local model folder (discovered by ModelManager).
- Supports txt2img and img2img (best-effort using pipeline's arguments).
- Supports basic face-preserve (face-lock) via insightface (creates mask from face bbox).
- Exposes performance toggles: dtype (fp16/bf16/float32), enable_model_cpu_offload.

This plugin is defensive: it reports clear errors when required libraries are missing.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("engine.flux")

# Lazy imports
_torch = None
_diffusers = None
_optimum = None
_PIL = None
_insightface = None

try:
    import torch as _torch
except Exception:
    _torch = None

try:
    from PIL import Image as _PILImage
    _PIL = _PILImage
except Exception:
    _PIL = None

try:
    import diffusers as _diffusers
except Exception:
    _diffusers = None

try:
    from optimum.quanto import quantize as _quantize, qfloat8 as _qfloat8, freeze as _freeze
    _optimum = True
except Exception:
    _optimum = False

try:
    import insightface as _insightface
except Exception:
    _insightface = None


@dataclass
class LoadOptions:
    torch_dtype: Optional[str] = None  # 'fp16','bf16','float32'
    enable_model_cpu_offload: bool = True
    device: str = "cuda"  # or 'cpu'


class EnginePlugin:
    """Engine plugin for FLUX models.

    Usage:
        plugin = EnginePlugin()
        plugin.load(model_path_or_id)
        plugin.run(request)
    """

    def __init__(self):
        self.name = "flux"
        self.model_path: Optional[Path] = None
        self.pipeline = None
        self.loaded = False
        self.options = LoadOptions(torch_dtype="fp16", enable_model_cpu_offload=True, device="cuda")
        self._face_detector = None

    def _ensure_dependencies(self) -> None:
        missing = []
        if _torch is None:
            missing.append("torch")
        if _diffusers is None:
            missing.append("diffusers")
        if _PIL is None:
            missing.append("Pillow")
        if missing:
            raise RuntimeError(f"Missing dependencies for FLUX engine: {', '.join(missing)}. Install them before loading the engine.")

    def _ensure_face_detector(self):
        global _insightface
        if _insightface is None:
            raise RuntimeError("insightface is not installed — face-preserve (face-lock) requires insightface")
        if self._face_detector is None:
            try:
                # Prepare a face analysis instance. ctx_id=0 for CUDA, -1 for CPU
                ctx = 0 if self.options.device.startswith("cuda") and _torch and _torch.cuda.is_available() else -1
                app = _insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider','CPUExecutionProvider'])
                app.prepare(ctx_id=0, det_size=(640, 640))
                self._face_detector = app
            except Exception as e:
                logger.warning("Failed to initialize insightface FaceAnalysis: %s", e)
                raise

    def _create_face_mask(self, image: "_PILImage.Image") -> "_PILImage.Image":
        """Detect first face in the image and create a binary mask where face region=255 (white) and background=0.
        This is a simple bounding-box based mask with padding. For better results, use landmarks and segmentation.
        """
        if _insightface is None:
            raise RuntimeError("insightface not available for face mask generation")
        self._ensure_face_detector()
        import numpy as _np

        img = _np.array(image.convert("RGB"))
        faces = self._face_detector.get(img)
        if not faces:
            raise RuntimeError("No faces detected in reference image")
        face = faces[0]
        # bbox
        x1, y1, x2, y2 = map(int, face.bbox.astype(int))
        # expand bbox a bit
        w = x2 - x1
        h = y2 - y1
        pad_w = int(0.25 * w)
        pad_h = int(0.25 * h)
        xa = max(0, x1 - pad_w)
        ya = max(0, y1 - pad_h)
        xb = min(image.width, x2 + pad_w)
        yb = min(image.height, y2 + pad_h)
        mask = _PIL.new("L", (image.width, image.height), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([xa, ya, xb, yb], fill=255)
        return mask

    def _find_model_path(self, model_id_or_path: str | Path) -> Optional[Path]:
        # Accept either full path or folder name in models dir
        p = Path(model_id_or_path)
        if p.exists():
            return p
        # search common models dir (respect AI_STUDIO_MODELS_PATH env or ./models)
        env = os.environ.get("AI_STUDIO_MODELS_PATH")
        candidates = []
        if env:
            candidates.append(Path(env) / model_id_or_path)
        candidates.append(Path("./models") / model_id_or_path)
        candidates.append(Path(model_id_or_path))
        for c in candidates:
            if c.exists():
                return c
        return None

    def load(self, model_id_or_path: str | Path, *, dtype: Optional[str] = None, device: Optional[str] = None, cpu_offload: Optional[bool] = None) -> None:
        """Load the FLUX model from disk (folder containing scheduler, transformer, vae, tokenizers...)

        model_id_or_path: folder name or absolute path
        dtype: 'fp16','bf16','float32' or None (default from options)
        device: 'cuda' or 'cpu'
        cpu_offload: enable model CPU offload (bool)
        """
        self._ensure_dependencies()
        if dtype:
            self.options.torch_dtype = dtype
        if device:
            self.options.device = device
        if cpu_offload is not None:
            self.options.enable_model_cpu_offload = cpu_offload

        model_path = self._find_model_path(model_id_or_path)
        if model_path is None:
            raise FileNotFoundError(f"Model path not found for '{model_id_or_path}'. Please place model folder in models/ or set AI_STUDIO_MODELS_PATH environment variable.")
        self.model_path = model_path
        logger.info("Loading FLUX model from %s", self.model_path)

        # Try to load components
        try:
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler
            from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
            from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
            from diffusers import AutoencoderKL
        except Exception as e:
            logger.exception("Required libraries for FLUX pipeline not available: %s", e)
            raise

        # choose torch_dtype
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32, None: None}
        torch_dtype = dtype_map.get(self.options.torch_dtype, None)
        # Load scheduler
        scheduler = None
        try:
            sched_path = self.model_path / "scheduler"
            if sched_path.exists():
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self.model_path / "scheduler"))
            else:
                # try load from parent hub by folder name
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self.model_path))
        except Exception:
            logger.warning("Failed loading scheduler from model folder, falling back to default scheduler")
            try:
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="scheduler")
            except Exception:
                scheduler = None

        # Load text encoders / tokenizers / vae / transformer — best-effort from local folder
        try:
            text_encoder = CLIPTextModel.from_pretrained(str(self.model_path / "text_encoder"), torch_dtype=torch_dtype) if (self.model_path / "text_encoder").exists() else CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch_dtype)
            tokenizer = CLIPTokenizer.from_pretrained(str(self.model_path / "tokenizer")) if (self.model_path / "tokenizer").exists() else CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

            text_encoder_2 = None
            tokenizer_2 = None
            if (self.model_path / "text_encoder_2").exists():
                text_encoder_2 = T5EncoderModel.from_pretrained(str(self.model_path / "text_encoder_2"), torch_dtype=torch_dtype)
            if (self.model_path / "tokenizer_2").exists():
                tokenizer_2 = T5TokenizerFast.from_pretrained(str(self.model_path / "tokenizer_2"))

            vae = AutoencoderKL.from_pretrained(str(self.model_path / "vae"), torch_dtype=torch_dtype) if (self.model_path / "vae").exists() else None

            transformer = None
            if (self.model_path / "transformer").exists():
                from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
                transformer = FluxTransformer2DModel.from_pretrained(str(self.model_path / "transformer"), torch_dtype=torch_dtype)
        except Exception as e:
            logger.exception("Error loading model components: %s", e)
            raise

        # Create pipeline
        try:
            pipe_kwargs: Dict[str, Any] = {
                "scheduler": scheduler,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "vae": vae,
                "transformer": transformer,
            }
            if text_encoder_2 is not None:
                pipe_kwargs["text_encoder_2"] = text_encoder_2
            if tokenizer_2 is not None:
                pipe_kwargs["tokenizer_2"] = tokenizer_2

            pipe = FluxPipeline(**{k: v for k, v in pipe_kwargs.items() if v is not None})

            # Set dtype/device
            if torch_dtype is not None:
                try:
                    pipe.to(torch_dtype)
                except Exception:
                    logger.debug("Unable to call pipe.to(dtype) — will rely on components' dtype")

            if self.options.enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    logger.debug("enable_model_cpu_offload not supported or failed")

            # Keep pipeline
            self.pipeline = pipe
            self.loaded = True
            logger.info("FLUX pipeline loaded successfully")
        except Exception as e:
            logger.exception("Failed to build FluxPipeline: %s", e)
            raise

    def _save_image(self, image: "_PIL.Image", prefix: str = "output") -> str:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        import time
        timestamp = f"{time.time():.7f}".replace('.', '')
        filename = f"{prefix}_{timestamp}.png"
        path = out_dir / filename
        image.save(path, format="PNG")
        return str(path)

    def run(self, request: "Any") -> Dict[str, Any]:
        """Run a generation request.

        The request is expected to have attributes similar to GenerateRequest from PipelineManager:
          prompt, seed, steps, width, height, guidance_scale, init_image (optional), strength, extra
        """
        if not self.loaded or self.pipeline is None:
            raise RuntimeError("Engine not loaded. Call load(model_path) first.")

        # Map request fields safely
        prompt = getattr(request, "prompt", "")
        seed = getattr(request, "seed", None)
        steps = getattr(request, "steps", 28)
        width = getattr(request, "width", 1024)
        height = getattr(request, "height", 1024)
        guidance = getattr(request, "extra", {}).get("guidance_scale", getattr(request, "extra", {}).get("cfg_scale", getattr(request, "guidance_scale", 3.5)))
        guidance = getattr(request, "guidance_scale", guidance)
        strength = getattr(request, "strength", 0.8)

        init_image_path = getattr(request, "init_image", None)
        face_lock = getattr(request, "extra", {}).get("face_lock", False)

        # Set generator
        generator = None
        if seed is not None and _torch is not None:
            try:
                generator = _torch.Generator(device=self.options.device).manual_seed(int(seed))
            except Exception:
                try:
                    generator = _torch.Generator().manual_seed(int(seed))
                except Exception:
                    generator = None

        # If no init_image => txt2img
        pil_image = None
        try:
            if init_image_path:
                if isinstance(init_image_path, str):
                    init_image = _PIL.open(init_image_path).convert("RGB")
                else:
                    init_image = init_image_path  # could be PIL image
                # Prepare mask if face_lock
                mask = None
                if face_lock:
                    try:
                        mask = self._create_face_mask(init_image)
                    except Exception as e:
                        logger.warning("Face lock requested but failed to create mask: %s", e)
                        mask = None
                # Call pipeline for img2img/inpainting if supported
                call_kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "generator": generator,
                    "guidance_scale": guidance,
                }
                # size args
                call_kwargs.setdefault("width", width)
                call_kwargs.setdefault("height", height)
                if mask is not None:
                    # many diffusers pipelines accept mask_image arg
                    call_kwargs["mask_image"] = mask
                # some pipelines accept image/init_image or init_image
                # Try common variants
                result = None
                for img_arg in ("image", "init_image", "init_images"):
                    try:
                        call_kwargs[img_arg] = init_image
                        out = self.pipeline(**call_kwargs)
                        result = out
                        break
                    except TypeError:
                        # try next
                        call_kwargs.pop(img_arg, None)
                    except Exception as e:
                        logger.exception("Pipeline img2img call failed: %s", e)
                        raise
                if result is None:
                    # fallback: run txt2img ignoring init image
                    logger.info("Pipeline did not accept init_image parameter — falling back to txt2img")
                    out = self.pipeline(prompt=prompt, width=width, height=height, num_inference_steps=steps, generator=generator, guidance_scale=guidance)
                else:
                    out = result
            else:
                # txt2img
                out = self.pipeline(prompt=prompt, width=width, height=height, num_inference_steps=steps, generator=generator, guidance_scale=guidance)

            # Extract image
            images = getattr(out, "images", None)
            if images is None and isinstance(out, dict):
                images = out.get("images")
            if images:
                pil_image = images[0]
            else:
                # some pipelines return a PIL image directly
                if isinstance(out, _PIL.Image.Image):
                    pil_image = out
                else:
                    # try to detect
                    if hasattr(out, 'images'):
                        pil_image = out.images[0]

            if pil_image is None:
                raise RuntimeError("Pipeline did not return an image")

            # Save output
            saved = self._save_image(pil_image, prefix="flux_out")
            logger.info("Generation completed, saved to %s", saved)
            return {"status": "ok", "output": saved}
        except Exception as e:
            logger.exception("Generation failed: %s", e)
            return {"status": "error", "reason": str(e)}

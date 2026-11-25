"""Model loading and initialization for Chatterbox TTS."""
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

import torch
from chatterbox.tts import ChatterboxTTS, REPO_ID
from huggingface_hub import hf_hub_download

_LOGGER = logging.getLogger(__name__)

# The files ChatterboxTTS.from_pretrained fetches; from_local expects all of them
MODEL_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
)


@lru_cache(maxsize=1)
def load_model(model_dir: Path, device: str = "cuda", dtype: str = "float32"):
    """Load Chatterbox TTS model (cached)."""
    _LOGGER.info(f"Loading Chatterbox TTS model on {device} with dtype {dtype}...")

    # Patch watermarker if perth module is not available or broken
    _patch_watermarker()

    # Load model
    model = ChatterboxTTS.from_local(ensure_weights(model_dir), device)
    
    # Convert to specified dtype if not float32
    if dtype.lower() not in ["float32", "fp32"]:
        _convert_model_dtype(model, dtype)
    
    # Warmup for cudagraphs (recommended by Chatterbox docs)
    if device == "cuda":
        _warmup_model(model)
    
    _LOGGER.info("Model loaded successfully")
    return model


def ensure_weights(model_dir: Path) -> Path:
    """Download any model files missing from model_dir and return it.

    Files already present are used as-is and never re-checked against the hub,
    so a populated directory loads with no network access.
    """
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.ismount(model_dir):
        _LOGGER.warning(
            f"{model_dir} is not a mounted volume; model weights will be "
            f"re-downloaded whenever the container is recreated"
        )

    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if not missing:
        _LOGGER.info(f"Using model weights from {model_dir}")
        return model_dir

    # hf_hub_download stages each file under model_dir/.cache and moves it into
    # place when complete, so an interrupted download never leaves a partial file
    # that the is_file() check above would accept
    for name in missing:
        _LOGGER.info(f"Downloading {name} from {REPO_ID} into {model_dir}...")
        hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=model_dir)

    return model_dir


def _patch_watermarker():
    """Patch watermarker if perth module is not available or broken."""
    try:
        import perth
        if perth.PerthImplicitWatermarker is None:
            raise ImportError("Perth watermarker not properly initialized")
    except (ImportError, AttributeError) as e:
        _LOGGER.warning(f"Perth watermarking not available ({e}), disabling watermarking")
        # Create a dummy watermarker class that mimics the Perth API
        class DummyWatermarker:
            def apply_watermark(self, audio, *args, **kwargs):
                """Pass-through watermark that returns audio unchanged."""
                return audio
            
            def __call__(self, audio, *args, **kwargs):
                return audio
        
        # Monkey-patch it into the module
        import sys
        if 'perth' not in sys.modules:
            import types
            sys.modules['perth'] = types.ModuleType('perth')
        sys.modules['perth'].PerthImplicitWatermarker = lambda: DummyWatermarker()


def _convert_model_dtype(model, dtype: str):
    """Convert model to specified dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    
    if torch_dtype:
        _LOGGER.info(f"Converting model to {torch_dtype}...")
        # Use the recommended t3_to pattern from Chatterbox docs
        model.t3.to(dtype=torch_dtype)
        model.conds.t3.to(dtype=torch_dtype)
        torch.cuda.empty_cache()
        _LOGGER.info(f"Model converted to {torch_dtype}")


def _warmup_model(model):
    """Warmup model with cudagraphs optimization."""
    _LOGGER.info("Warming up model with cudagraphs optimization...")
    # First warmup call sets up the cudagraph
    model.generate(
        "Warmup for cudagraphs optimization.", 
        t3_params={"generate_token_backend": "cudagraphs-manual"}
    )
    # Second call runs at full speed
    model.generate(
        "Second warmup for full cudagraphs speed.",
        t3_params={"generate_token_backend": "cudagraphs-manual"}
    )
    _LOGGER.info("Warmup complete - cudagraphs ready")


# Serialises every mutation of model.conds. Both conditioning and generation
# assign it, and on-demand conditioning runs while requests are in flight.
# Reentrant because the Wyoming handler re-enters generation to fall back to a
# default voice while already holding the lock.
MODEL_LOCK = threading.RLock()

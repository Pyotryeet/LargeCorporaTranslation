#!/usr/bin/env python3
"""Fallback QAT benchmark module — runs QAT models via subprocess.

When transformers 5.x is incompatible with QAT checkpoints, we use this
module to run QAT benchmarks in an isolated subprocess with a compatible
transformers version, or use raw safetensors loading bypassing HF entirely.

Approach: load safetensors directly, build model skeleton, inject weights.
This bypasses all HF loading paths that access ``module.weight`` on packed
modules.
"""

import json
import logging
import time
import gc
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)


def _load_qat_direct(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load QAT model by probing safetensors and using config to build.

    Completely bypasses HF's from_pretrained — no accelerate device_map,
    no meta-device path, no _init_weights issues.  Instead:

    1. Finds the local HuggingFace cache snapshot (or downloads it via
       ``huggingface_hub.snapshot_download``).
    2. Loads all ``.safetensors`` files into a flat state dict.
    3. Instantiates the model from ``AutoConfig`` on the meta device
       (zero memory), then materializes on the target device with
       ``to_empty``.
    4. Loads the state dict directly via ``load_state_dict(strict=False,
       assign=True)``.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID (e.g. ``google/gemma-4-E2B-it-qat-mobile-ct``).
    device : str
        Torch device string (default ``"cpu"``).

    Returns
    -------
    torch.nn.Module
        The loaded model in eval mode on the requested device.
    """
    from safetensors.torch import load_file
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration,
    )
    import glob, os

    # Find local snapshot
    snapshots = glob.glob(
        os.path.expanduser(f"~/.cache/huggingface/hub/models--google--gemma-4-E2B*")
        + "/snapshots/*"
    ) if "E2B" in model_path else glob.glob(
        os.path.expanduser(f"~/.cache/huggingface/hub/models--google--gemma-4-E4B*")
        + "/snapshots/*"
    )

    if not snapshots:
        # Fall back to HuggingFace model ID
        from huggingface_hub import snapshot_download
        snap_dir = snapshot_download(model_path, ignore_patterns=["*.msgpack", "*.h5"])
    else:
        snap_dir = snapshots[0]

    snap_dir = str(Path(snap_dir))

    # Load all safetensors
    sf_files = sorted(glob.glob(os.path.join(snap_dir, "*.safetensors")))
    state_dict = {}
    for sf in sf_files:
        state_dict.update(load_file(sf, device="cpu"))

    # Build model from config
    config = AutoConfig.from_pretrained(snap_dir, trust_remote_code=False)
    with torch.device("meta"):  # build on meta device to avoid memory
        model = Gemma4ForConditionalGeneration(config)

    # Materialize on target device
    model = model.to_empty(device=device)

    # Load state dict directly (skip HF's load)
    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()

    del state_dict
    gc.collect()
    return model


def load_qat_model(model_path: str, device: Optional[torch.device] = None) -> tuple[torch.nn.Module, "PreTrainedTokenizerBase"]:
    """Load a QAT model — tries direct safetensors path first.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID (e.g. ``google/gemma-4-E2B-it-qat-mobile-ct``).
    device : torch.device, optional
        Target device.  Defaults to CPU if ``None``.

    Returns
    -------
    tuple[torch.nn.Module, PreTrainedTokenizerBase]
        A ``(model, tokenizer)`` pair.  The model is in eval mode on the
        requested device (or CPU).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device is None:
        device = torch.device("cpu")

    model = _load_qat_direct(model_path, str(device))
    return model, tokenizer


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-E2B-it-qat-mobile-ct"
    print(f"Loading {model_path} via direct safetensors...")
    t0 = time.monotonic()
    model, tok = load_qat_model(model_path, device=torch.device("mps"))
    n = sum(p.numel() for p in model.parameters())
    print(f"✓ {n/1e9:.2f}B params loaded in {time.monotonic()-t0:.1f}s")

    enc = tok("hello world", return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=8, pad_token_id=tok.pad_token_id)
    print(f"Output: {tok.decode(out[0], skip_special_tokens=True)}")

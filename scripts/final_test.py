import os, sys, time
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from benchmark.hardware.backend import detect_backend
from benchmark.inference.engine import InferenceEngine
from benchmark.inference.sampling import DecodingParams
from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import TextChunker
from benchmark.data.filters import ChunkFilter
from benchmark.data.pipeline import AsyncPipeline

device_info = detect_backend("cuda")
print(f"2x {device_info.name}")

engine = InferenceEngine(
    model_path="mistralai/Ministral-3B-Instruct", tokenizer_path="",
    device_info=device_info,
    decoding_params=DecodingParams(max_new_tokens=64, temperature=0.0),
    use_flash_attention=True, use_torch_compile=True,
    max_input_tokens=512, backend_type="auto",
)
engine.load()
be = engine._backend
fp8_method = getattr(be, "_fp8_method", "?")
fp8_active = getattr(be, "_fp8_active", False)
print(f"FP8: method={fp8_method} active={fp8_active}")
engine._configured_batch_size = 32
print("Warming up...")
engine.warmup(batches=10)
print("Warmup OK!")

loader = JSONLLoader(["data/input/*.jsonl.gz"], shuffle=False)
chunker = TextChunker(engine.tokenizer, 512, 50)
filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)
pipeline = AsyncPipeline(loader, chunker, engine.tokenizer, filt,
                         batch_size=32, prefetch_workers=4, backend="cuda")
pipeline.start_prefetch()

batches, tokens = 0, 0
t0 = time.monotonic()
while time.monotonic() - t0 < 60:
    batch = pipeline.next_batch()
    if batch is None:
        if pipeline.draining():
            break
        continue
    result = engine.translate(batch)
    pipeline.release_batch(batch)
    batches += 1
    tokens += result.output_tokens_total

elapsed = time.monotonic() - t0
tps = tokens / elapsed if elapsed > 0 else 0
pipeline.stop_prefetch()
print("")
print("=" * 50)
print(f"NGC CONTAINER — TE FP8 + COMPILE + SDPA + TF32")
print(f"  Model: Ministral 3B | bs=32 | 1xH200")
print(f"  {batches} batches, {tokens} tok, {elapsed:.1f}s")
print(f"  {tps:.0f} tok/s")
print("=" * 50)

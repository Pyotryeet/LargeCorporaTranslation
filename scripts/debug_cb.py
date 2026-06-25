import os, sys, time
sys.path.insert(0, '.')
os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
os.environ['TR_SKIP_FP8'] = '1'
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('google/translategemma-4b-it', trust_remote_code=False)
print(f'Config type: {type(cfg).__name__}')
for attr in ['hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'num_hidden_layers', 'text_config']:
    try:
        val = getattr(cfg, attr, 'MISSING')
        if attr == 'text_config' and val != 'MISSING':
            for a2 in ['hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'num_hidden_layers']:
                print(f'  text_config.{a2} = {getattr(val, a2, "MISSING")}')
        else:
            print(f'  {attr} = {val}')
    except Exception as e:
        print(f'  {attr}: ERROR {e}')

# Now test CB with correct values
from benchmark.hardware.backend import detect_backend
from benchmark.inference.engine import InferenceEngine
from benchmark.inference.sampling import DecodingParams
from benchmark.inference.continuous_batcher import ContinuousBatcher
from benchmark.inference.paged_attention import PagedKVCache
from benchmark.data.pretokenizer import ensure_pretokenized

device_info = detect_backend('cuda')
engine = InferenceEngine(
    model_path='google/translategemma-4b-it', tokenizer_path='',
    device_info=device_info,
    decoding_params=DecodingParams(max_new_tokens=32, temperature=0.0),
    use_flash_attention=True, use_torch_compile=False,
    max_input_tokens=512, backend_type='auto',
)
engine.load()
engine._configured_batch_size = 16
engine.warmup(batches=5)

# Get correct kv config
be = engine._backend
kv_cfg = be.kv_cache_config
cfg = engine.model.config
tc = getattr(cfg, 'text_config', cfg)
head_dim = getattr(tc, 'head_dim', None) or getattr(cfg, 'head_dim', None) or (getattr(tc, 'hidden_size', 2560) // getattr(tc, 'num_attention_heads', 16))
num_layers = getattr(tc, 'num_hidden_layers', None) or getattr(cfg, 'num_hidden_layers', 36)
num_kv_heads = getattr(tc, 'num_key_value_heads', None) or getattr(cfg, 'num_key_value_heads', 4)
print(f'\nkv_cache_config: {kv_cfg}')
print(f'head_dim={head_dim} layers={num_layers} kv_heads={num_kv_heads}')

pretok = ensure_pretokenized('google/translategemma-4b-it', engine.tokenizer, max_input_tokens=512, input_paths=['data/input/*.jsonl.gz'])

paged_kv = PagedKVCache(
    num_layers=num_layers, num_kv_heads=num_kv_heads,
    head_dim=head_dim, block_size=16, num_blocks=1024,
    dtype=torch.bfloat16, device=engine.devices[0],
)
batcher = ContinuousBatcher(
    engine, paged_kv, max_batch_size=16,
    pad_token_id=engine.tokenizer.pad_token_id or 0,
)

# Process 3 chunks
t0 = time.monotonic()
count = 0
for text, token_ids, tok_count in pretok.iter_chunks():
    if count >= 3:
        break
    ids = torch.tensor([token_ids], dtype=torch.long, device=engine.devices[0])
    batcher.submit(ids, text)
    count += 1

print(f'\nSubmitted {count} sequences. Running: {batcher.running_count()} Waiting: {batcher.waiting_count()}')

step = 0
while batcher.running_count() > 0 or batcher.waiting_count() > 0:
    completed = batcher.step()
    step += 1
    if step % 20 == 0:
        elapsed = time.monotonic() - t0
        print(f'  step {step}: running={batcher.running_count()} waiting={batcher.waiting_count()} elapsed={elapsed:.1f}s')
    if step > 500:
        print(f'  ABORT at step {step}')
        break

elapsed = time.monotonic() - t0
print(f'Done in {elapsed:.1f}s, {step} steps')
for seq in batcher.flush_completed():
    text = engine.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
    print(f'  Output ({len(seq.generated_ids)} tok): {text[:120]}...')

del engine; torch.cuda.empty_cache()

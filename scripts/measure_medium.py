"""Medium measurements: M1.3 (batch ceiling), M2.6 (PagedAttn memory), M2.7 (quant speedup)."""
import torch, os, gc, time, json
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

RESULTS = {}
print("=" * 60)
print("MEDIUM MEASUREMENT SWEEP")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ===== M1.3: Batch-Size Ceiling (binary search) =====
print("\n--- M1.3: Batch-Size Ceiling (TranslateGemma 4B, BF16) ---")
model_id = 'google/translategemma-4b-it'
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
cfg._attn_implementation = 'sdpa'
model = AutoModelForCausalLM.from_pretrained(
    model_id, config=cfg, torch_dtype=torch.bfloat16,
    device_map='cuda:0', trust_remote_code=False,
)
model.eval()
tok = AutoTokenizer.from_pretrained(model_id)
mem_alloc = torch.cuda.memory_allocated(0)/1024**3
mem_total = torch.cuda.get_device_properties(0).total_memory/1024**3
print(f"  Model VRAM: {mem_alloc:.1f} GB / {mem_total:.1f} GB total")

# Binary search for OOM boundary
low, high = 1, 512
oom_boundary = None
results = []
while low <= high:
    bs = (low + high) // 2
    try:
        text = f'Test sentence for batch sizing.' * 3
        batch = [text] * bs
        inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to('cuda:0')
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or 0)
        # Success — try larger
        low = bs + 1
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        high = bs - 1
        oom_boundary = bs
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            high = bs - 1
            oom_boundary = bs
        else:
            raise
        torch.cuda.empty_cache()

max_viable = high
safety_85 = int(max_viable * 0.85)
mem_peak = torch.cuda.max_memory_allocated(0)/1024**3
print(f"  OOM boundary: ~{oom_boundary}")
print(f"  Max viable:   {max_viable}")
print(f"  Safety 85%:   {safety_85}")
print(f"  Peak mem:     {mem_peak:.1f} GB")
print(f"  Tuner default safety_margin: 0.15 -> batch = {int(max_viable * 0.85)}")

# Measure TPS at key sizes to find optimal
print(f"  Throughput at key sizes:")
for bs in sorted(set([1, 4, 8, 16, 32, 64, 128, safety_85, max_viable])):
    if bs > max_viable: continue
    texts = [f'Sentence {i} for batch throughput measurement at size {bs}.' for i in range(bs * 4)]
    total_out = 0
    try:
        torch.cuda.synchronize(); start = time.time()
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda:0')
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or 0)
            total_out += (out.shape[1] - inputs.input_ids.shape[1]) * len(batch)
        torch.cuda.synchronize(); duration = time.time() - start
        tps = total_out / duration
        print(f"    bs={bs:>3d}: {tps:>8.1f} tok/s")
        results.append({'bs': bs, 'tps': round(tps, 1)})
    except:
        print(f"    bs={bs:>3d}: OOM/error")
        break
    finally:
        torch.cuda.empty_cache()

optimal = max(results, key=lambda r: r['tps'])
print(f"  OPTIMAL: bs={optimal['bs']} at {optimal['tps']:.1f} tok/s")
print(f"  Current tuner picks: bs={safety_85} (85% of OOM)")
if optimal['bs'] != safety_85:
    print(f"  MISMATCH: optimal={optimal['bs']}, safety={safety_85} — tuner leaves performance on the table!")
else:
    print(f"  Tuner is optimal ✓")

RESULTS['M1.3'] = {
    'oom_boundary': oom_boundary, 'max_viable': max_viable,
    'safety_85': safety_85, 'optimal_bs': optimal['bs'],
    'optimal_tps': optimal['tps'], 'safety_tps': next((r['tps'] for r in results if r['bs'] == safety_85), None),
}

del model; gc.collect(); torch.cuda.empty_cache()

# ===== M2.7: Weight Quantization Speedup (INT8, 4B) =====
print("\n--- M2.7: Weight Quantization Speedup ---")
for quant, desc in [('bf16', 'BF16 baseline'), ('int8', 'INT8 (bitsandbytes)')]:
    try:
        torch.cuda.empty_cache(); gc.collect()
        if quant == 'bf16':
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map='cuda:0',
                trust_remote_code=False,
            )
        else:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb, device_map='cuda:0',
                trust_remote_code=False,
            )
        model.eval()
        mem = torch.cuda.memory_allocated(0)/1024**3
        texts = [f'Sentence {i} for throughput at {quant}.' for i in range(64)]
        total_out = 0; bs = 8
        torch.cuda.synchronize(); start = time.time()
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda:0')
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or 0)
            total_out += (out.shape[1] - inputs.input_ids.shape[1]) * len(batch)
        torch.cuda.synchronize(); duration = time.time() - start
        tps = total_out / duration
        print(f"  {desc}: {mem:.1f} GB, {tps:.1f} tok/s")
        RESULTS[f'M2.7_{quant}'] = {'memory_gb': round(mem, 1), 'tps': round(tps, 1)}
        del model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {desc}: FAILED — {str(e)[:100]}")

# ===== M2.6: PagedAttention Memory Savings =====
print("\n--- M2.6: PagedAttention Memory Savings ---")
from benchmark.config.constants import PAGED_BLOCK_SIZE
# Theoretical: continuous KV cache vs paged blocks
# For a batch of 8, seq_len 512:
num_layers = 34; num_kv_heads = 4; head_dim = 256; bytes_per_elem = 2
seq_len = 512
batch_size = 8
# Continuous: batch * 2 * layers * kv_heads * head_dim * seq_len * bytes
continuous_bytes = batch_size * 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_elem
# Paged: round up to blocks (16 tokens per block)
blocks_needed = (seq_len + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
paged_bytes = batch_size * 2 * num_layers * num_kv_heads * head_dim * blocks_needed * PAGED_BLOCK_SIZE * bytes_per_elem
# Actual paged overhead: the paged tensor includes unused tail slots
overhead_pct = ((paged_bytes - continuous_bytes) / continuous_bytes) * 100
savings_pct = ((continuous_bytes - paged_bytes) / continuous_bytes) * 100
print(f"  Continuous KV (bs=8, seq=512): {continuous_bytes/1024**3:.2f} GB")
print(f"  Paged KV (blocks={blocks_needed}×{PAGED_BLOCK_SIZE}): {paged_bytes/1024**3:.2f} GB")
print(f"  Overhead (paged > continuous): {overhead_pct:.1f}%")
print(f"  Savings (paged < continuous): {savings_pct:.1f}%")
print(f"  Note: The '40-70%' claim applies to overallocation scenarios")
print(f"  (variable seq lengths, pre-allocation of max_len). At fixed seq_len,")
print(f"  paged has overhead due to block alignment.")
# The savings come from NOT pre-allocating max_seq_len for every sequence
max_seq_len = 4096
cont_max = batch_size * 2 * num_layers * num_kv_heads * head_dim * max_seq_len * bytes_per_elem
blocks_at_512 = (512 + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
paged_at_512 = batch_size * 2 * num_layers * num_kv_heads * head_dim * blocks_at_512 * PAGED_BLOCK_SIZE * bytes_per_elem
actual_savings = ((cont_max - paged_at_512) / cont_max) * 100
print(f"  Real scenario (pre-alloc 4096, actual 512):")
print(f"    Continuous (alloc 4096): {cont_max/1024**3:.2f} GB")
print(f"    Paged (alloc as needed):  {paged_at_512/1024**3:.2f} GB")
print(f"    REAL SAVINGS: {actual_savings:.1f}%")
RESULTS['M2.6'] = {'continuous_alloc_4096_gb': round(cont_max/1024**3, 2), 'paged_as_needed_gb': round(paged_at_512/1024**3, 2), 'savings_pct': round(actual_savings, 1)}

# ===== SAVE =====
with open('/tmp/medium_measurements.json', 'w') as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nDONE. Measurements: {list(RESULTS.keys())}")

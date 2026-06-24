import subprocess
"""M0.5 + M4.2: Throughput degradation over extended run (4+ hours)."""
import torch, os, gc, time, json, sys
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

HOURS = 4
SAVE_INTERVAL = 300  # save every 5 minutes

print(f"M0.5/M4.2 Degradation Test — {HOURS}h run")
print(f"Start: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_id = 'google/translategemma-4b-it'
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
cfg._attn_implementation = 'sdpa'
model = AutoModelForCausalLM.from_pretrained(
    model_id, config=cfg, torch_dtype=torch.bfloat16,
    device_map='cuda:0', trust_remote_code=False,
)
model.eval()
tok = AutoTokenizer.from_pretrained(model_id)
mem = torch.cuda.memory_allocated(0) / 1024**3
print(f"Model loaded: {mem:.1f} GB VRAM")

# Generate diverse batch texts (cycle through them for the whole run)
base_texts = [
    f'Translate English to Turkish: This is document number {i} in our corpus. It contains various English sentences about technology, science, and daily life.' for i in range(200)
]
bs = 16

# Warmup
for _ in range(5):
    batch = base_texts[:bs]
    inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda:0')
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=32, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or 0)
torch.cuda.empty_cache()
print(f"Warmup complete. Starting {HOURS}h measurement...")

samples = []  # list of (elapsed_seconds, tokens, duration, tps)
total_out = 0
batch_count = 0
start_time = time.time()
last_save = start_time
save_path = '/tmp/degradation_data.jsonl'

try:
    end_time = start_time + HOURS * 3600
    text_idx = 0
    while time.time() < end_time:
        batch_start = time.time()
        texts = []
        for _ in range(bs):
            texts.append(base_texts[text_idx % len(base_texts)])
            text_idx += 1

        inputs = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda:0')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or 0)
        tokens = (out.shape[1] - inputs.input_ids.shape[1]) * len(texts)
        total_out += tokens
        batch_count += 1

        elapsed = time.time() - start_time
        batch_dur = time.time() - batch_start
        tps = tokens / batch_dur if batch_dur > 0 else 0
        samples.append((elapsed, tokens, batch_dur, tps))

        # Save periodically
        if time.time() - last_save > SAVE_INTERVAL:
            with open(save_path, 'a') as f:
                for s in samples[-batch_count:]:
                    f.write(json.dumps({'elapsed_s': round(s[0], 1), 'tokens': s[1], 'duration_s': round(s[2], 3), 'tps': round(s[3], 1)}) + '\n')
            mem_now = torch.cuda.memory_allocated(0)/1024**3
            gpu_temp = subprocess.run(['nvidia-smi', '--query-gpu=index,temperature.gpu,power.draw,clocks.current.sm', '--format=csv,noheader'], capture_output=True, text=True)
            temps = gpu_temp.stdout.strip()
            hours_done = elapsed / 3600
            recent_tps = sum(s[3] for s in samples[-20:]) / min(20, len(samples)) if samples else 0
            print(f'  [{hours_done:.1f}h] batches={batch_count} tokens={total_out:,} recent_tps={recent_tps:.0f} mem={mem_now:.1f}GB GPU:{temps}')
            last_save = time.time()

    run_duration = time.time() - start_time
except KeyboardInterrupt:
    run_duration = time.time() - start_time
    print(f"Interrupted after {run_duration:.1f}s")

# Final save
with open(save_path, 'a') as f:
    for s in samples[-batch_count:]:
        f.write(json.dumps({'elapsed_s': round(s[0], 1), 'tokens': s[1], 'duration_s': round(s[2], 3), 'tps': round(s[3], 1)}) + '\n')

# Degradation analysis
import numpy as np
elapsed = np.array([s[0] for s in samples])
tps_vals = np.array([s[3] for s in samples])

# Linear regression
coeffs = np.polyfit(elapsed, tps_vals, 1)
slope = coeffs[0]  # TPS change per second
slope_per_hour = slope * 3600
mean_tps = tps_vals.mean()

# R-squared
residuals = tps_vals - np.polyval(coeffs, elapsed)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((tps_vals - mean_tps)**2)
r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# Segment analysis: compare first 30min vs last 30min
n = len(samples)
first_third = tps_vals[:n//3].mean() if n >= 3 else tps_vals.mean()
last_third = tps_vals[2*n//3:].mean() if n >= 3 else tps_vals.mean()
degradation_pct = ((last_third - first_third) / first_third) * 100

print(f"\n{'='*60}")
print(f"DEGRADATION RESULTS")
print(f"{'='*60}")
print(f"Duration:          {run_duration/3600:.1f} hours")
print(f"Batches:           {batch_count}")
print(f"Total tokens:      {total_out:,}")
print(f"Mean TPS:          {mean_tps:.1f}")
print(f"Slope:             {slope_per_hour:.2f} tok/s per hour ({slope_per_hour/mean_tps*100:.2f}%/hr)")
print(f"R-squared:         {r_squared:.4f}")
print(f"First 1/3 mean:    {first_third:.1f} tok/s")
print(f"Last 1/3 mean:     {last_third:.1f} tok/s")
print(f"Degradation:        {degradation_pct:.1f}%")
print(f"Is degrading:       {slope_per_hour < -0.01 * mean_tps and r_squared > 0.1}")
print(f"Data saved:        {save_path}")

del model; gc.collect(); torch.cuda.empty_cache()
print("\nDONE.")



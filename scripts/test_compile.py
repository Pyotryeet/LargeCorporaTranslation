"""Test torch.compile on PyTorch 2.6.0 + H200 SM90."""
import torch, os, gc, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
torch.cuda.empty_cache(); gc.collect()

print('=== Test: torch.compile (reduce-overhead) ===')
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    'google/translategemma-4b-it',
    torch_dtype=torch.bfloat16, device_map='cuda:0',
    trust_remote_code=False,
)
model.eval()
tok = AutoTokenizer.from_pretrained('google/translategemma-4b-it')

inputs = tok('Translate English to Turkish: Hello', return_tensors='pt').to('cuda:0')
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or 0)

print('Applying torch.compile(reduce-overhead)...')
compile_works = False
try:
    model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    inputs2 = tok('Hello world test', return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        out2 = model.generate(**inputs2, max_new_tokens=20, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or 0)
    print(f'SUCCESS: {tok.decode(out2[0], skip_special_tokens=True)[:100]}')
    compile_works = True
except Exception as e:
    print(f'FAILED: {str(e)[:300]}')

if compile_works:
    texts = [f'Translate English to Turkish: Speed test number {i}. Measuring throughput.' for i in range(40)]
    print('Compiled throughput:')
    for bs in [4, 8, 16]:
        total_out = 0
        torch.cuda.synchronize(); start = time.time()
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda:0')
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=48, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or 0)
            total_out += (out.shape[1] - inputs.input_ids.shape[1]) * len(batch)
        torch.cuda.synchronize(); duration = time.time() - start
        tps = total_out / duration
        print(f'  bs={bs:>3d}: {tps:>8.1f} tok/s  ({total_out} out tokens in {duration:.1f}s)')

del model; gc.collect(); torch.cuda.empty_cache()
print(f'\ntorch.compile: {"WORKS" if compile_works else "FAILED"}')

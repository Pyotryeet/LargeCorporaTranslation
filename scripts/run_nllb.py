"""Benchmark NLLB models on fixed PyTorch 2.6.0+cu124."""
import torch, os, gc, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print('NLLB Throughput (Flash SDPA, BF16, PyTorch 2.6.0+cu124)')
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

for model_id, name in [
    ('facebook/nllb-200-distilled-600M', 'NLLB-200 600M'),
    ('facebook/nllb-200-distilled-1.3B', 'NLLB-200 1.3B'),
    ('facebook/nllb-200-3.3B', 'NLLB-200 3.3B'),
]:
    print(f'\n--- {name} ---')
    torch.cuda.empty_cache(); gc.collect()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map='cuda:0',
        trust_remote_code=False,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    mem = torch.cuda.memory_allocated(0)/1024**3
    params = sum(p.numel() for p in model.parameters())/1e9
    print(f'  {params:.2f}B params, {mem:.1f} GB VRAM')
    
    # Warmup
    tok.src_lang = 'eng_Latn'
    for _ in range(2):
        inputs = tok('Hello world', return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids('tur_Latn'), max_new_tokens=10, num_beams=1)
    
    # Throughput
    texts = [f'This is sentence number {i} used for translation throughput testing of the benchmark harness.' for i in range(200)]
    total_in = 0; total_out = 0
    bs = 8
    torch.cuda.synchronize(); start = time.time()
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        tok.src_lang = 'eng_Latn'
        inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda:0')
        total_in += int(inputs.attention_mask.sum().item())
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids('tur_Latn'), max_new_tokens=48, num_beams=1)
        total_out += out.shape[1] * len(batch)
    torch.cuda.synchronize(); duration = time.time() - start
    tps = total_out / duration
    ips = total_in / duration
    print(f'  bs=8: {tps:.1f} tok/s output, {ips:.1f} tok/s input ({total_out} out in {duration:.1f}s)')
    
    del model; gc.collect(); torch.cuda.empty_cache()

print('\nDONE.')

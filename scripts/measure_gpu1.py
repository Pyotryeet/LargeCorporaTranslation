"""Remaining measurements on GPU 1: M2.7 INT8, M1.5 TE FP8, M2.2 CB, M2.3 speculative."""
import torch, os, gc, time, json
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# GPU 1 is selected by caller via CUDA_VISIBLE_DEVICES=1

print("GPU 1 SWEEP")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"Visible devices: {torch.cuda.device_count()} → {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
model_id = 'google/translategemma-4b-it'
tok = AutoTokenizer.from_pretrained(model_id)

# ===== M2.7: INT8 Quantization Speedup =====
print("\n=== M2.7: INT8 Weight Quantization Speedup ===")
results = {}
for config_name, load_kwargs in [
    ('BF16', {'torch_dtype': torch.bfloat16}),
    ('INT8', {'quantization_config': BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)}),
]:
    torch.cuda.empty_cache(); gc.collect()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map='cuda:0', trust_remote_code=False, **load_kwargs)
        model.eval()
        mem = torch.cuda.memory_allocated(0)/1024**3
        params = sum(p.numel() for p in model.parameters())
        # Warmup
        for _ in range(3):
            inputs = tok('Hello world', return_tensors='pt').to('cuda:0')
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                   num_beams=1, pad_token_id=tok.pad_token_id or 0)
        # Throughput
        texts = [f'Sentence {i} for INT8 throughput measurement at batch size 16.' for i in range(64)]
        total_out = 0; bs = 16
        torch.cuda.synchronize(); start = time.time()
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda:0')
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                     num_beams=1, pad_token_id=tok.pad_token_id or 0)
            total_out += (out.shape[1] - inputs.input_ids.shape[1]) * len(batch)
        torch.cuda.synchronize(); duration = time.time() - start
        tps = total_out / duration
        mem_alloc = torch.cuda.memory_allocated(0)/1024**3
        mem_reserved = torch.cuda.memory_reserved(0)/1024**3
        print(f"  {config_name}: {mem_alloc:.1f}GB alloc, {mem_reserved:.1f}GB reserved, {tps:.1f} tok/s")
        results[config_name] = {'mem_alloc_gb': round(mem_alloc, 1), 'mem_reserved_gb': round(mem_reserved, 1),
                                'tps': round(tps, 1), 'params': params}
        del model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {config_name}: FAILED — {str(e)[:200]}")
        results[config_name] = {'error': str(e)[:200]}

if 'BF16' in results and 'INT8' in results and 'tps' in results['INT8']:
    mem_savings = (1 - results['INT8']['mem_alloc_gb'] / results['BF16']['mem_alloc_gb']) * 100
    tps_change = (results['INT8']['tps'] / results['BF16']['tps'] - 1) * 100
    print(f"  INT8 vs BF16: {mem_savings:.0f}% memory savings, {tps_change:+.0f}% TPS change")

# ===== M1.5: TE FP8 (smoke test — TE 2.16 warns about torch < 2.11) =====
print("\n=== M1.5: TE FP8 Smoke Test ===")
torch.cuda.empty_cache(); gc.collect()
try:
    import transformer_engine
    print(f"  TE version: {transformer_engine.__version__}")
    from transformer_engine.pytorch import fp8_autocast
    print(f"  fp8_autocast available")

    # Try FP8 via TE on a simple model
    from benchmark.hardware.precision import apply_te_fp8_to_model
    torch.cuda.empty_cache(); gc.collect()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map='cuda:0', trust_remote_code=False)
    model.eval()
    mem_before = torch.cuda.memory_allocated(0)/1024**3
    try:
        apply_te_fp8_to_model(model, skip_lm_head=True)
        mem_after = torch.cuda.memory_allocated(0)/1024**3
        # Test generate
        inputs = tok('Hello world', return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            with fp8_autocast(enabled=True):
                out = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                                    num_beams=1, pad_token_id=tok.pad_token_id or 0)
        decoded = tok.decode(out[0], skip_special_tokens=True)[:80]
        print(f"  TE FP8 applied: {mem_before:.1f}GB → {mem_after:.1f}GB ({((mem_after-mem_before)/mem_before)*100:+.0f}%)")
        print(f"  Generate: {decoded}")
        # Quick TPS
        texts = [f'Sentence {i} for FP8 throughput test.' for i in range(48)]
        total_out = 0; bs = 16
        torch.cuda.synchronize(); start = time.time()
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda:0')
            with torch.no_grad():
                with fp8_autocast(enabled=True):
                    out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                        num_beams=1, pad_token_id=tok.pad_token_id or 0)
            total_out += (out.shape[1] - inputs.input_ids.shape[1]) * len(batch)
        torch.cuda.synchronize(); duration = time.time() - start
        fp8_tps = total_out / duration
        results['TE_FP8'] = {'mem_before_gb': round(mem_before,1), 'mem_after_gb': round(mem_after,1),
                              'tps': round(fp8_tps, 1), 'status': 'working'}
        print(f"  TE FP8 TPS: {fp8_tps:.1f} tok/s")
    except Exception as e:
        print(f"  TE FP8 forward: FAILED — {str(e)[:200]}")
        results['TE_FP8'] = {'status': 'failed', 'error': str(e)[:200]}
    del model; gc.collect(); torch.cuda.empty_cache()
except ImportError as e:
    print(f"  TE not importable: {str(e)[:100]}")
    results['TE_FP8'] = {'status': 'not_installed'}
except Exception as e:
    print(f"  TE setup: FAILED — {str(e)[:200]}")
    results['TE_FP8'] = {'status': 'error', 'error': str(e)[:200]}

# ===== M2.3: Speculative Decoding (GPU 1) =====
print("\n=== M2.3: Speculative Decoding ===")
os.environ['TR_ENABLE_EXPERIMENTAL_SPECULATIVE'] = '1'
torch.cuda.empty_cache(); gc.collect()
try:
    from benchmark.inference.speculative import create_speculative_decoder
    from benchmark.inference.backends.autoregressive import AutoregressiveBackend
    from benchmark.inference.backends.protocol import BackendConfig
    from benchmark.hardware.backend import DeviceInfo
    import torch as _t
    device_info = DeviceInfo(backend='cuda', device=_t.device('cuda:0'), num_devices=1, name='NVIDIA H200 NVL')
    backend_config = BackendConfig(
        model_path='google/translategemma-4b-it', tokenizer_path='google/translategemma-4b-it',
        device_info=device_info, max_input_tokens=512, max_new_tokens=512,
        dtype='auto', use_flash_attention=True, use_torch_compile=False,
        extra={'use_speculative': True, 'speculative_mode': 'self',
               'speculative_num_tokens': 3, 'speculative_num_draft_layers': 0},
    )
    backend = AutoregressiveBackend(backend_config)
    backend.load()
    if backend._spec_decoder is not None:
        backend._configured_batch_size = 1  # spec is serial per-sequence
        # Warmup
        texts = ['Translate English to Turkish: Hello world.']
        for _ in range(3):
            from benchmark.data.pipeline import PipelineBatch
            import numpy as np
            inputs = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=64).to('cuda:0')
            pb = type('PipelineBatch', (), {
                'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask,
                'raw_texts': texts, 'batch_id': 0,
            })()
            result = backend.translate_batch(pb)
        # Throughput (speculative, bs=1, greedy)
        texts = [f'Translate English to Turkish: This is test sentence number {i} used for benchmarking speculative decoding throughput.' for i in range(100)]
        total_out = 0
        torch.cuda.synchronize(); start = time.time()
        for text in texts:
            inputs = tok([text], return_tensors='pt', padding=True, truncation=True, max_length=64).to('cuda:0')
            pb = type('PipelineBatch', (), {
                'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask,
                'raw_texts': [text], 'batch_id': 0,
            })()
            result = backend.translate_batch(pb)
            total_out += result.output_tokens_total
        torch.cuda.synchronize(); duration = time.time() - start
        spec_tps = total_out / duration
        spec_stats = backend._spec_decoder.stats
        print(f"  Speculative (bs=1): {spec_tps:.1f} tok/s, acceptance={spec_stats.get('acceptance_rate', '?'):.2f}")
        results['M2.3_speculative'] = {'tps': round(spec_tps, 1), 'acceptance_rate': spec_stats.get('acceptance_rate', 'unknown'),
                                        'drafted': spec_stats.get('total_drafted', 0), 'accepted': spec_stats.get('total_accepted', 0)}
        backend.close()
    else:
        print(f"  Speculative decoder returned None — env var not respected?")
        results['M2.3_speculative'] = {'status': 'not_created'}
    del backend; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    print(f"  Speculative: FAILED — {str(e)[:300]}")
    results['M2.3_speculative'] = {'status': 'error', 'error': str(e)[:300]}

# ===== SAVE =====
with open('/tmp/gpu1_measurements.json', 'w') as f:
    f.write(json.dumps(results, indent=2))
print(f"\n=== DONE. Results: {list(results.keys())} ===")

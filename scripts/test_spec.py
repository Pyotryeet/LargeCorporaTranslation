"""M2.3: Test speculative decoding with dual-RoPE fix."""
import torch, os, time, gc, types
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TR_ENABLE_EXPERIMENTAL_SPECULATIVE'] = '1'
print(f'torch {torch.__version__}')

from benchmark.inference.backends.autoregressive import AutoregressiveBackend
from benchmark.inference.backends.protocol import BackendConfig
from benchmark.hardware.backend import DeviceInfo

di = DeviceInfo(backend='cuda', device=torch.device('cuda:0'), num_devices=1, name='H200')
bc = BackendConfig(
    model_path='google/translategemma-4b-it',
    tokenizer_path='google/translategemma-4b-it',
    device_info=di, max_input_tokens=512, max_new_tokens=512,
    dtype='auto', use_flash_attention=True, use_torch_compile=False,
    extra={'use_speculative': True, 'speculative_mode': 'self',
           'speculative_num_tokens': 3, 'speculative_num_draft_layers': 0},
)
be = AutoregressiveBackend(bc)
be.load()

if be._spec_decoder:
    sd = be._spec_decoder
    print(f'Spec LOADED: draft={sd._num_draft_layers}/{sd._total_layers} layers')

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('google/translategemma-4b-it')

    # Warmup
    for _ in range(3):
        txt = ['Translate English to Turkish: Hello world.']
        inp = tok(txt, return_tensors='pt', padding=True, truncation=True, max_length=64).to('cuda:0')
        pb = types.SimpleNamespace(input_ids=inp.input_ids, attention_mask=inp.attention_mask, raw_texts=txt, batch_id=0)
        _ = sd.translate_batch(pb, be)

    # Throughput (serial, bs=1)
    texts = [f'Translate English to Turkish: Speed test {i}. Spec decode.' for i in range(40)]
    total_out = 0
    torch.cuda.synchronize(); start = time.time()
    for txt in texts:
        inp = tok([txt], return_tensors='pt', padding=True, truncation=True, max_length=64).to('cuda:0')
        pb = types.SimpleNamespace(input_ids=inp.input_ids, attention_mask=inp.attention_mask, raw_texts=[txt], batch_id=0)
        result = sd.translate_batch(pb, be)
        total_out += result.output_tokens_total
    torch.cuda.synchronize(); dur = time.time() - start
    tps = total_out / dur
    st = sd.stats
    print(f'Spec TPS: {tps:.0f} tok/s, accept={st.get("acceptance_rate",0):.2f}, drafted={st.get("total_drafted",0)}, accepted={st.get("total_accepted",0)}')
else:
    print('Spec NOT LOADED')

be.close(); del be; gc.collect(); torch.cuda.empty_cache()
print('M2.3 DONE')

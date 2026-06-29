#!/usr/bin/env python3
"""Translate 30 selected sentences with all 5 models.

Corrected from web research:
  - NLLB:     src_lang in tokenizer CONSTRUCTOR, max_length=512, num_beams=4
  - MADLAD:   T5ForConditionalGeneration + T5Tokenizer (NOT Auto classes),
              <2tr> prefix, max_new_tokens=256, num_beams=4
  - Gemma:    chat template, eos=<end_of_turn> (106), do_sample=False

Usage:  python3 scripts/evaluation/translate_sentences.py
"""
import json, gc, time, os
from pathlib import Path
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration, T5Tokenizer,
)

ROOT = Path(__file__).resolve().parent.parent.parent
INFILE = ROOT / "data" / "output" / "model_selection" / "source_sentences.json"
OUTFILE = ROOT / "data" / "output" / "model_selection" / "translations.json"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DTYPE = torch.bfloat16 if DEVICE in ("mps", "cuda") else torch.float32

with open(INFILE) as f:
    sentences = json.load(f)
print(f"Device: {DEVICE}  Sentences: {len(sentences)}")
output = [{"source_id": s["id"], "source_text": s["text"],
           "source_char_len": s["char_len"], "models": {}} for s in sentences]

# ════════════════════════════════════════════════════════════════
# NLLB — src_lang in constructor, max_length=512, num_beams=4
# ════════════════════════════════════════════════════════════════
for mid, mpath in [
    ("nllb_600m", "facebook/nllb-200-distilled-600M"),
    ("nllb_1.3b", "facebook/nllb-200-distilled-1.3B"),
    ("nllb_3.3b", "facebook/nllb-200-3.3B"),
    ("nllb_moe_54b", "facebook/nllb-moe-54b"),
]:
    print(f"\n=== {mid} ===")
    tok = AutoTokenizer.from_pretrained(mpath, src_lang="eng_Latn")
    device_map = "auto" if "moe" in mpath else None
    m = AutoModelForSeq2SeqLM.from_pretrained(
        mpath, torch_dtype=DTYPE, trust_remote_code=False,
        device_map=device_map,
    ).eval()
    if device_map != "auto":
        m = m.to(DEVICE)
    for e in output:
        t0 = time.time()
        lines = e["source_text"].split("\n")
        translated_lines = []
        total_tokens = 0
        for line in lines:
            if not line.strip():
                translated_lines.append("")
                continue
            inp = tok(line, return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                out = m.generate(**inp,
                    forced_bos_token_id=tok.convert_tokens_to_ids("tur_Latn"),
                    max_length=512, num_beams=4)
            translated_line = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
            import html
            translated_lines.append(html.unescape(translated_line))
            total_tokens += out.shape[1]
        text = "\n".join(translated_lines)
        e["models"][mid] = {"text": text,
            "output_tokens": total_tokens,
            "latency_ms": round((time.time()-t0)*1000, 1)}
    del m; gc.collect()

# ════════════════════════════════════════════════════════════════
# MADLAD — T5ForConditionalGeneration + T5Tokenizer, <2tr> prefix
# ════════════════════════════════════════════════════════════════
for mid, mpath in [
    ("madlad_3b", "google/madlad400-3b-mt"),
    ("madlad_10b", "google/madlad400-10b-mt"),
]:
    print(f"\n=== {mid} ===")
    tok = T5Tokenizer.from_pretrained(mpath)
    device_map = "auto" if "10b" in mpath else None
    m = T5ForConditionalGeneration.from_pretrained(
        mpath, torch_dtype=DTYPE,
        device_map=device_map,
    ).eval()
    if device_map != "auto":
        m = m.to(DEVICE)
    # Tie embedding weights manually to resolve scale mismatch and gibberish outputs
    m.shared.weight = m.decoder.embed_tokens.weight
    m.encoder.embed_tokens.weight = m.decoder.embed_tokens.weight

    for e in output:
        inp = tok(f"<2tr> {e['source_text']}", return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        t0 = time.time()
        with torch.no_grad():
            out = m.generate(input_ids=inp["input_ids"],
                             max_new_tokens=200, no_repeat_ngram_size=3)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        e["models"][mid] = {"text": text,
            "output_tokens": out.shape[1],
            "latency_ms": round((time.time()-t0)*1000, 1)}
    del m; gc.collect()

# ════════════════════════════════════════════════════════════════
# TranslateGemma — chat template, eos=<end_of_turn>
# ════════════════════════════════════════════════════════════════
for mid, mpath in [
    ("translategemma_4b", "google/translategemma-4b-it"),
    ("translategemma_12b", "google/translategemma-12b-it"),
    ("translategemma_27b", "google/translategemma-27b-it"),
]:
    print(f"\n=== {mid} ===")
    tok = AutoTokenizer.from_pretrained(mpath)
    device_map = "auto" if "12b" in mpath or "27b" in mpath else None
    m = AutoModelForCausalLM.from_pretrained(
        mpath, torch_dtype=DTYPE, trust_remote_code=False,
        device_map=device_map,
    ).eval()
    if device_map != "auto":
        m = m.to(DEVICE)
    for e in output:
        prompt = (
            f"<start_of_turn>user\n"
            f"Translate the following English text to Turkish. "
            f"Do not include any explanations, introduction, markdown formatting, or surrounding quote marks. "
            f"Strictly output only the clean translation text.\n\n"
            f"Text to translate:\n{e['source_text']}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        inp = tok(prompt, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        t0 = time.time()
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=256, do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=[tok.eos_token_id, 106])
        text = tok.batch_decode(
            out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0].strip()
        import html
        text = html.unescape(text)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        e["models"][mid] = {"text": text,
            "output_tokens": (out[0] != tok.pad_token_id).sum().item() - int(inp["input_ids"].shape[1]),
            "latency_ms": round((time.time()-t0)*1000, 1)}
    del m; gc.collect()

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTFILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nDONE — {len(output)} sources × {len(output[0]['models'])} models → {OUTFILE}")

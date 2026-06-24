#!/usr/bin/env python3
"""Translate 30 selected sentences with 6 candidate models.

Loads source_sentences.json, runs all 6 models on all 30 sentences
using the existing inference backend infrastructure, and saves
translations to translations.json.

Models (5 — SmolLM2 dropped: instruct model, prompt leak, not a translator):
  1. facebook/nllb-200-distilled-600M  (encoder-decoder)
  2. facebook/nllb-200-distilled-1.3B  (encoder-decoder)
  3. facebook/nllb-200-3.3B            (encoder-decoder)
  4. google/madlad400-3b-mt            (encoder-decoder)
  5. google/translategemma-4b-it       (autoregressive)

Output: data/output/model_selection/translations.json
"""
import json, os, time, gc
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "output" / "model_selection" / "source_sentences.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "model_selection" / "translations.json"

# ── 6 candidate models ──────────────────────────────────────────────────
MODELS = [
    {
        "id": "nllb_600m",
        "name": "NLLB-200 600M",
        "path": "facebook/nllb-200-distilled-600M",
        "backend_type": "encoder_decoder",
        "family": "nllb",
        "src_lang": "eng_Latn",
        "tgt_lang": "tur_Latn",
    },
    {
        "id": "nllb_1.3b",
        "name": "NLLB-200 1.3B",
        "path": "facebook/nllb-200-distilled-1.3B",
        "backend_type": "encoder_decoder",
        "family": "nllb",
        "src_lang": "eng_Latn",
        "tgt_lang": "tur_Latn",
    },
    {
        "id": "nllb_3.3b",
        "name": "NLLB-200 3.3B",
        "path": "facebook/nllb-200-3.3B",
        "backend_type": "encoder_decoder",
        "family": "nllb",
        "src_lang": "eng_Latn",
        "tgt_lang": "tur_Latn",
    },
    {
        "id": "madlad_3b",
        "name": "MADLAD-400 3B",
        "path": "google/madlad400-3b-mt",
        "backend_type": "encoder_decoder",
        "family": "madlad",
        "src_lang": "eng_Latn",
        "tgt_lang": "tur_Latn",
    },
    {
        "id": "translategemma_4b",
        "name": "TranslateGemma 4B",
        "path": "google/translategemma-4b-it",
        "backend_type": "autoregressive",
        "family": "ar",
    },
    {
        "id": "smollm2_1.7b",
        "name": "SmolLM2 1.7B",
        "path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "backend_type": "autoregressive",
        "family": "ar",
    },
]


def translate_nllb(model_id, model_path, sentences, src_lang, tgt_lang):
    """Translate using NLLB encoder-decoder backend."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.cuda.empty_cache(); gc.collect()
    print(f"    Loading {model_path}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=False,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    results = []
    for sent in sentences:
        tok.src_lang = src_lang
        inputs = tok(sent["text"], return_tensors="pt", truncation=True, max_length=256).to("cuda:0")
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(tgt_lang),
                max_new_tokens=256, num_beams=1, early_stopping=False,
            )
        latency_ms = (time.time() - start) * 1000
        output_ids = out[0][inputs.input_ids.shape[1]:]
        text = tok.decode(output_ids, skip_special_tokens=True).strip()
        results.append({
            "text": text,
            "output_tokens": len(output_ids),
            "latency_ms": round(latency_ms, 1),
        })
    del model; gc.collect(); torch.cuda.empty_cache()
    return results


def translate_madlad(model_path, sentences):
    """Translate using MADLAD encoder-decoder (no src_lang needed, uses <2tr> prefix)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.cuda.empty_cache(); gc.collect()
    print(f"    Loading {model_path}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=False,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    results = []
    for sent in sentences:
        # MADLAD uses explicit <2tr> prefix
        prompt = f"<2tr> {sent['text']}"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda:0")
        start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, num_beams=1)
        latency_ms = (time.time() - start) * 1000
        output_ids = out[0][inputs.input_ids.shape[1]:]
        text = tok.decode(output_ids, skip_special_tokens=True).strip()
        results.append({
            "text": text,
            "output_tokens": len(output_ids),
            "latency_ms": round(latency_ms, 1),
        })
    del model; gc.collect(); torch.cuda.empty_cache()
    return results


def translate_autoregressive(model_path, sentences):
    """Translate using autoregressive backend (TranslateGemma or SmolLM2)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.empty_cache(); gc.collect()
    print(f"    Loading {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=False,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    results = []
    for sent in sentences:
        prompt = f"Translate English to Turkish:\n{sent['text']}"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda:0")
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id or 0,
            )
        latency_ms = (time.time() - start) * 1000
        output_ids = out[0][inputs.input_ids.shape[1]:]
        text = tok.decode(output_ids, skip_special_tokens=True).strip()
        results.append({
            "text": text,
            "output_tokens": len(output_ids),
            "latency_ms": round(latency_ms, 1),
        })
    del model; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    # Load sentences
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = json.load(f)
    print(f"Loaded {len(sentences)} source sentences")

    output = []
    for sent in sentences:
        output.append({
            "source_id": sent["id"],
            "source_text": sent["text"],
            "source_char_len": sent["char_len"],
            "models": {},
        })

    for model_def in MODELS:
        mid = model_def["id"]
        mname = model_def["name"]
        mpath = model_def["path"]
        family = model_def["family"]
        print(f"\n{'='*60}")
        print(f"Model: {mname} ({mid}) — {family}")
        print(f"{'='*60}")

        try:
            if family == "nllb":
                results = translate_nllb(
                    mid, mpath, sentences,
                    model_def["src_lang"], model_def["tgt_lang"],
                )
            elif family == "madlad":
                results = translate_madlad(mpath, sentences)
            elif family == "ar":
                results = translate_autoregressive(mpath, sentences)
            else:
                print(f"  Unknown family: {family}")
                continue

            # Store results
            for i, r in enumerate(results):
                output[i]["models"][mid] = r

            # Show samples
            print(f"  Translated {len(results)} sentences")
            for j in range(min(3, len(results))):
                t = results[j]["text"][:80]
                print(f"    [{j}] {t}...")

        except Exception as e:
            print(f"  ERROR: {str(e)[:200]}")
            for i in range(len(sentences)):
                output[i]["models"][mid] = {
                    "text": f"[ERROR: {str(e)[:100]}]",
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "error": str(e)[:200],
                }

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved 30 sources × {len(MODELS)} models = "
          f"{len(output) * len(MODELS)} translations to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run automated quality metrics comparison between BF16 and FP8 precisions.

Evaluates:
  - All standard models: NLLB (600M, 1.3B, 3.3B), MADLAD 3B, TranslateGemma 4B (in BF16)
  - Big models: NLLB MoE 54B, MADLAD 10B, TranslateGemma 12B, TranslateGemma 27B
    in BOTH BF16 and FP8 (optimum-quanto float8) precisions.

Outputs:
  - data/output/model_selection/metrics_comparison.csv
  - data/output/model_selection/translations_comparison.json
"""
import json, gc, time, os, csv, html
from pathlib import Path
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration, T5Tokenizer,
)

# Apply COMET compatibility monkey patch for transformers 5.x
try:
    from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + [self.bos_token_id] + token_ids_1 + [self.eos_token_id]
    XLMRobertaTokenizer.build_inputs_with_special_tokens = _build_inputs_with_special_tokens
    XLMRobertaTokenizerFast.build_inputs_with_special_tokens = _build_inputs_with_special_tokens

    import comet.encoders.xlmr
    def _forward_patched(self, input_ids, attention_mask, **kwargs):
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return {
            "sentemb": model_output.last_hidden_state[:, 0, :],
            "wordemb": model_output.last_hidden_state,
            "all_layers": model_output.hidden_states,
            "attention_mask": attention_mask,
        }
    comet.encoders.xlmr.XLMREncoder.forward = _forward_patched
except Exception as e:
    print(f"Warning: Monkey patching failed: {e}")

ROOT = Path(__file__).resolve().parent.parent.parent
INFILE = ROOT / "data" / "output" / "model_selection" / "source_sentences.json"
REF_FILE = ROOT / "data" / "output" / "model_selection" / "references.jsonl"
OUT_CSV = ROOT / "data" / "output" / "model_selection" / "metrics_comparison.csv"
OUT_JSON = ROOT / "data" / "output" / "model_selection" / "translations_comparison.json"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE in ("mps", "cuda") else torch.float32

def load_references():
    """Load references as a dict keyed by source_text for proper alignment."""
    ref_map = {}
    if not REF_FILE.exists():
        return ref_map
    with open(REF_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                src = obj.get("source_text", "").strip()
                r = obj.get("reference_translation", "")
                if src and r:
                    ref_map[src] = r
            except Exception: pass
    return ref_map

_sp_tokenizer = None

def compute_spbleu(hyps, refs):
    global _sp_tokenizer
    import sacrebleu
    if _sp_tokenizer is None:
        try:
            _sp_tokenizer = AutoTokenizer.from_pretrained("google/translategemma-4b-it")
        except Exception:
            return 0.0
    sp_tok = _sp_tokenizer
    hyp_tok = [" ".join(sp_tok.convert_ids_to_tokens(sp_tok.encode(h, add_special_tokens=False))) for h in hyps]
    ref_tok = [[" ".join(sp_tok.convert_ids_to_tokens(sp_tok.encode(r, add_special_tokens=False))) for r in refs]]
    result = sacrebleu.corpus_bleu(hyp_tok, ref_tok, tokenize="none")
    return round(result.score, 1)

def compute_morph_bleu(hyps, refs):
    import sacrebleu
    TR_SUFFIXES = [
        "ler", "lar", "de", "da", "den", "dan", "e", "a", "i", "u", "ü", "ı",
        "in", "un", "ün", "ın", "nin", "nun", "nün", "nın",
        "dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür",
        "li", "lı", "lu", "lü", "siz", "sız", "suz", "süz",
        "ce", "ca", "çe", "ça", "ki", "kü", "ci", "cı", "cu", "cü",
        "miş", "mış", "muş", "müş", "di", "dı", "du", "dü",
    ]
    TR_SUFFIXES.sort(key=len, reverse=True)
    def strip_suffixes(word):
        lower = word.lower()
        for s in TR_SUFFIXES:
            if lower.endswith(s) and len(lower) > len(s) + 2:
                return lower[:-len(s)]
        return lower
    hyp_stripped = [" ".join(strip_suffixes(w) for w in h.split()) for h in hyps]
    ref_stripped = [[" ".join(strip_suffixes(w) for w in r.split()) for r in refs]]
    result = sacrebleu.corpus_bleu(hyp_stripped, ref_stripped, tokenize="intl")
    return round(result.score, 1)

def main():
    print(f"Starting comparison evaluation on device: {DEVICE}...")
    with open(INFILE) as f:
        sentences = json.load(f)
    ref_map = load_references()
    sources = [s["text"] for s in sentences]
    # Build aligned refs list matching source order
    refs = [ref_map.get(s.strip(), "") for s in sources]
    n_matched = sum(1 for r in refs if r)
    print(f"  Refs matched: {n_matched}/{len(sources)}")
    
    results = [] # list of dicts for CSV export
    translations_log = [] # detailed log of translations

    # Define model run configuration
    # (model_id, path, is_large, model_type)
    models_config = [
        ("nllb_600m", "facebook/nllb-200-distilled-600M", False, "nllb"),
        ("nllb_1.3b", "facebook/nllb-200-distilled-1.3B", False, "nllb"),
        ("nllb_3.3b", "facebook/nllb-200-3.3B", False, "nllb"),
        ("nllb_moe_54b", "facebook/nllb-moe-54b", True, "nllb"),
        ("madlad_3b", "google/madlad400-3b-mt", False, "madlad"),
        ("madlad_10b", "google/madlad400-10b-mt", True, "madlad"),
        ("translategemma_4b", "google/translategemma-4b-it", False, "gemma"),
        ("translategemma_12b", "google/translategemma-12b-it", True, "gemma"),
        ("translategemma_27b", "google/translategemma-27b-it", True, "gemma"),
    ]

    for model_id, mpath, is_large, mtype in models_config:
        precisions = ["BF16", "FP8"] if (is_large and DEVICE == "cuda") else ["BF16"]
        
        for precision in precisions:
            print(f"\n================ Running model: {model_id} ({precision}) ================")
            t_start = time.time()
            hyps = []
            
            device_map = "auto" if is_large else None

            try:
                # 1. Load Tokenizer & Model
                if mtype == "nllb":
                    tok = AutoTokenizer.from_pretrained(mpath, src_lang="eng_Latn")
                    m = AutoModelForSeq2SeqLM.from_pretrained(
                        mpath, torch_dtype=DTYPE, trust_remote_code=False,
                        device_map=device_map
                    ).eval()
                    if device_map is None:
                        m = m.to(DEVICE)
                elif mtype == "madlad":
                    tok = T5Tokenizer.from_pretrained(mpath)
                    m = T5ForConditionalGeneration.from_pretrained(
                        mpath, torch_dtype=DTYPE,
                        device_map=device_map
                    ).eval()
                    if device_map is None:
                        m = m.to(DEVICE)
                    # Manually tie weights
                    m.shared.weight = m.decoder.embed_tokens.weight
                    m.encoder.embed_tokens.weight = m.decoder.embed_tokens.weight
                else: # gemma
                    tok = AutoTokenizer.from_pretrained(mpath)
                    m = AutoModelForCausalLM.from_pretrained(
                        mpath, torch_dtype=DTYPE, trust_remote_code=False,
                        device_map=device_map
                    ).eval()
                    if device_map is None:
                        m = m.to(DEVICE)

                # 2. SmoothQuant Calibration & Static FP8 Quantization
                if precision == "FP8":
                    from quantization.smoothquant import SmoothQuantCalibrator
                    from benchmark.hardware.precision import apply_static_fp8_to_model
                    print("Running SmoothQuant calibration...")
                    if tok.pad_token_id is None:
                        tok.pad_token = tok.eos_token
                    calibration_texts = [s["text"] for s in sentences]
                    calibrator = SmoothQuantCalibrator(m, tok, alpha=0.5, device=DEVICE)
                    calibrator.calibrate(calibration_texts)
                    print("Applying static FP8 quantization...")
                    apply_static_fp8_to_model(m, skip_lm_head=True)

                # 2. Run Inference
                latencies = []
                for s in sentences:
                    src_text = s["text"]
                    t0 = time.time()
                    
                    if mtype == "nllb":
                        lines = src_text.split("\n")
                        translated_lines = []
                        for line in lines:
                            if not line.strip():
                                translated_lines.append("")
                                continue
                            inp = tok(line, return_tensors="pt").to(DEVICE)
                            with torch.no_grad():
                                out = m.generate(**inp,
                                    forced_bos_token_id=tok.convert_tokens_to_ids("tur_Latn"),
                                    max_length=512, num_beams=4)
                            decoded = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
                            translated_lines.append(html.unescape(decoded))
                        text = "\n".join(translated_lines)
                    elif mtype == "madlad":
                        lines = src_text.split("\n")
                        translated_lines = []
                        for line in lines:
                            if not line.strip():
                                translated_lines.append("")
                                continue
                            inp = tok(f"<2tr> {line}", return_tensors="pt").to(DEVICE)
                            with torch.no_grad():
                                out = m.generate(input_ids=inp["input_ids"],
                                                 max_new_tokens=200, no_repeat_ngram_size=3)
                            translated_line = tok.decode(out[0], skip_special_tokens=True).strip()
                            translated_lines.append(translated_line)
                        text = "\n".join(translated_lines)
                    else: # gemma
                        prompt = (
                            f"<start_of_turn>user\n"
                            f"Translate the following English text to Turkish. "
                            f"Do not include any explanations, introduction, markdown formatting, or surrounding quote marks. "
                            f"Strictly output only the clean translation text.\n\n"
                            f"Text to translate:\n{src_text}"
                            f"<end_of_turn>\n<start_of_turn>model\n"
                        )
                        inp = tok(prompt, return_tensors="pt").to(DEVICE)
                        with torch.no_grad():
                            out = m.generate(**inp, max_new_tokens=256, do_sample=False,
                                pad_token_id=tok.eos_token_id,
                                eos_token_id=[tok.eos_token_id, 106])
                        decoded = tok.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
                        decoded = html.unescape(decoded)
                        if decoded.startswith('"') and decoded.endswith('"'):
                            decoded = decoded[1:-1].strip()
                        text = decoded

                    latencies.append((time.time() - t0) * 1000)
                    hyps.append(text)

                avg_latency = sum(latencies) / len(latencies)
                print(f"Inference complete. Latency: {avg_latency:.1f} ms/sentence")

                # Delete model to free memory
                del m
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                gc.collect()

                # 3. Compute Metrics — only use matched source-reference pairs
                print("Computing metrics...")
                from benchmark.quality.metrics_comet import compute_comet, compute_comet_kiwi
                from benchmark.quality.metrics_metricx import compute_metricx
                from benchmark.quality.metrics_bertscore import compute_bertscore
                import sacrebleu

                # Build matched triplets
                m_src, m_hyp, m_ref = [], [], []
                for s, h, r in zip(sources, hyps, refs):
                    if r:
                        m_src.append(s)
                        m_hyp.append(h)
                        m_ref.append(r)
                print(f"  Using {len(m_ref)}/{len(sources)} matched pairs")

                # COMET-Kiwi (reference-free, uses all)
                kiwi = compute_comet_kiwi(sources, hyps).get("system_score")
                # COMET-22
                comet = compute_comet(m_src, m_hyp, m_ref).get("system_score")
                # MetricX-24
                metricx = compute_metricx(m_src, m_hyp, m_ref).get("system_score")
                # BERTScore
                bert = compute_bertscore(m_ref, m_hyp).get("system_score")
                # chrF++
                chrf = round(sacrebleu.corpus_chrf(m_hyp, [m_ref], word_order=2).score, 1)
                # spBLEU
                spbleu = compute_spbleu(m_hyp, m_ref)
                # Morph-BLEU
                morph_bleu = compute_morph_bleu(m_hyp, m_ref)

                # Append to CSV results
                results.append({
                    "Model": model_id,
                    "Precision": precision,
                    "COMET-22": comet,
                    "COMET-Kiwi": kiwi,
                    "MetricX-24": metricx,
                    "BERTScore": bert,
                    "chrF++": chrf,
                    "spBLEU": spbleu,
                    "Morph-BLEU": morph_bleu,
                    "Avg_Latency_ms": round(avg_latency, 1)
                })

                # Append to JSON translations log
                translations_log.append({
                    "model_id": model_id,
                    "precision": precision,
                    "translations": hyps
                })

            except Exception as e:
                print(f"Error evaluating model {model_id} ({precision}): {e}")
                import traceback
                traceback.print_exc()

    # Save CSV Results
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Model", "Precision", "COMET-22", "COMET-Kiwi", "MetricX-24", "BERTScore", "chrF++", "spBLEU", "Morph-BLEU", "Avg_Latency_ms"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Save JSON Translations
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(translations_log, f, indent=2, ensure_ascii=False)

    print(f"\nALL DONE! Results written to:\n  - CSV: {OUT_CSV}\n  - JSON: {OUT_JSON}")

if __name__ == "__main__":
    main()

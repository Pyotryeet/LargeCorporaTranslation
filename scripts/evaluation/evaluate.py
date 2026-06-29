#!/usr/bin/env python3
"""Compute 10 automated evaluation metrics for all 180 translation pairs.

Metrics:
  1. chrF       (word_order=0, char_order=4)
  2. chrF++     (word_order=2, char_order=4)
  3. BLEU       (SacreBLEU, tokenizer="intl")
  4. SacreBLEU  (same as BLEU — sacrebleu is the implementation)
  5. spBLEU     (SentencePiece tokenizer + sacrebleu)
  6. Morph-BLEU (Turkish suffix-stripping + sacrebleu)
  7. BLEURT     (bleurt-pytorch, lucadiliello/BLEURT-20-D12)
  8. COMET      (Unbabel/wmt22-comet-da, reference-based)
  9. COMET-Kiwi (Unbabel/wmt22-cometkiwi-da, reference-free)
 10. MetricX-24 (google/metricx-24-hybrid-large-v2p6, reference-based)

Output: data/output/model_selection/metrics.json
"""
import json, os, time, gc
from pathlib import Path
import torch
import numpy as np

# Monkey patch to fix transformers 5.x compatibility shims with COMET
try:
    from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + [self.bos_token_id] + token_ids_1 + [self.eos_token_id]
    XLMRobertaTokenizer.build_inputs_with_special_tokens = _build_inputs_with_special_tokens
    XLMRobertaTokenizerFast.build_inputs_with_special_tokens = _build_inputs_with_special_tokens

    import comet.encoders.xlmr
    def _forward_patched(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
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
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "output" / "model_selection" / "translations.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "model_selection" / "metrics.json"


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} source entries with translations")

    model_ids = list(data[0]["models"].keys())
    print(f"Models: {model_ids}")
    print(f"Total translation pairs: {len(data) * len(model_ids)}")

    # ── Prepare data structures ─────────────────────────────────────────
    all_metrics = []
    hyps_by_model = {mid: [] for mid in model_ids}

    # Extract all hypotheses and source texts
    for entry in data:
        src = entry["source_text"]
        for mid in model_ids:
            hyp = entry["models"][mid]["text"]
            hyps_by_model[mid].append(hyp)

    # ── 1 & 2. chrF and chrF++ ──────────────────────────────────────────
    print("\n--- chrF / chrF++ ---")
    from benchmark.quality.metrics_chrf import compute_chrf
    from sacrebleu import corpus_chrf as _raw_chrf

    chrf_scores: dict[str, float | None] = {}
    chrfpp_scores: dict[str, float | None] = {}

    for mid in model_ids:
        hyps = hyps_by_model[mid]
        refs = [[src] for src in [e["source_text"] for e in data]]
        # chrF (char-only)
        try:
            chrf_result = _raw_chrf(hyps, refs, word_order=0, char_order=4)
            chrf_scores[mid] = round(chrf_result.score, 1)
        except Exception:
            chrf_scores[mid] = None
        # chrF++ (word+char)
        try:
            chrfpp_result = _raw_chrf(hyps, refs, word_order=2, char_order=4)
            chrfpp_scores[mid] = round(chrfpp_result.score, 1)
        except Exception:
            chrfpp_scores[mid] = None
        # Per-segment chrF++
        try:
            seg_scores = [round(_raw_chrf([h], [r], word_order=2, char_order=4).score, 1)
                          for h, r in zip(hyps, [[e["source_text"]] for e in data])]
        except Exception:
            seg_scores = [None] * len(hyps)

        print(f"  {mid}: chrF={chrf_scores[mid]}, chrF++={chrfpp_scores[mid]}")

        for i in range(len(data)):
            all_metrics.append({
                "source_id": data[i]["source_id"],
                "model_id": mid,
                "chrf": chrf_scores[mid] if i == 0 else None,
                "chrf_pp": chrfpp_scores[mid] if i == 0 else None,
                "chrf_seg": seg_scores[i],
            })

    # ── 3 & 4. BLEU / SacreBLEU ─────────────────────────────────────────
    print("\n--- BLEU / SacreBLEU ---")
    bleu_scores: dict[str, float | None] = {}
    for mid in model_ids:
        hyps = hyps_by_model[mid]
        refs = [[e["source_text"]] for e in data]
        try:
            from benchmark.quality.metrics_bleu import compute_bleu
            bleu_result = compute_bleu(hyps, refs)
            bleu_scores[mid] = bleu_result.get("score")
        except Exception as e:
            print(f"  {mid}: BLEU ERROR — {str(e)[:100]}")
            bleu_scores[mid] = None
        if bleu_scores[mid] is not None:
            print(f"  {mid}: BLEU={bleu_scores[mid]:.1f}")

    # ── 5. spBLEU (SentencePiece tokenizer + sacrebleu) ─────────────────
    print("\n--- spBLEU ---")
    spbleu_scores: dict[str, float | None] = {}
    try:
        from transformers import AutoTokenizer
        sp_tok = AutoTokenizer.from_pretrained("google/translategemma-4b-it")
    except Exception as e:
        print(f"  spBLEU tokenizer load error: {e}")
        sp_tok = None

    for mid in model_ids:
        hyps = hyps_by_model[mid]
        refs = [[e["source_text"]] for e in data]
        if sp_tok is not None:
            try:
                from sacrebleu import corpus_bleu, BLEU
                hyp_tok = [" ".join(sp_tok.convert_ids_to_tokens(sp_tok.encode(h, add_special_tokens=False))) for h in hyps]
                ref_tok = [[" ".join(sp_tok.convert_ids_to_tokens(sp_tok.encode(r[0], add_special_tokens=False)))] for r in refs]
                spbleu = corpus_bleu(hyp_tok, ref_tok, tokenize="none")
                spbleu_scores[mid] = round(spbleu.score, 1)
            except Exception as e:
                print(f"  {mid}: spBLEU ERROR — {str(e)[:100]}")
                spbleu_scores[mid] = None
        else:
            spbleu_scores[mid] = None
        if spbleu_scores[mid] is not None:
            print(f"  {mid}: spBLEU={spbleu_scores[mid]:.1f}")

    # ── 6. Morph-BLEU (Turkish suffix stripping) ────────────────────────
    print("\n--- Morph-BLEU ---")
    morph_scores: dict[str, float | None] = {}
    TR_SUFFIXES = [
        "ler", "lar", "de", "da", "den", "dan", "e", "a", "i", "u", "ü", "ı",
        "in", "un", "ün", "ın", "nin", "nun", "nün", "nın",
        "dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür",
        "li", "lı", "lu", "lü", "siz", "sız", "suz", "süz",
        "ce", "ca", "çe", "ça", "ki", "kü", "ci", "cı", "cu", "cü",
        "miş", "mış", "muş", "müş", "di", "dı", "du", "dü",
    ]
    TR_SUFFIXES.sort(key=len, reverse=True)

    def strip_turkish_suffixes(word):
        lower = word.lower()
        for suffix in TR_SUFFIXES:
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                return lower[:-len(suffix)]
        return lower

    for mid in model_ids:
        hyps = hyps_by_model[mid]
        refs = [[e["source_text"]] for e in data]
        try:
            from sacrebleu import corpus_bleu
            hyp_stripped = [" ".join(strip_turkish_suffixes(w) for w in h.split()) for h in hyps]
            ref_stripped = [[" ".join(strip_turkish_suffixes(w) for w in r[0].split())] for r in refs]
            morph_bleu = corpus_bleu(hyp_stripped, ref_stripped, tokenize="intl")
            morph_scores[mid] = round(morph_bleu.score, 1)
        except Exception as e:
            print(f"  {mid}: Morph-BLEU ERROR — {str(e)[:100]}")
            morph_scores[mid] = None
        if morph_scores[mid] is not None:
            print(f"  {mid}: Morph-BLEU={morph_scores[mid]:.1f}")

    # ── 7. BLEURT (bleurt-pytorch) ──────────────────────────────────────
    print("\n--- BLEURT ---")
    bleurt_scores = {}
    try:
        from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
        torch.cuda.empty_cache(); gc.collect()
        print("  Loading BLEURT-20-D12 model...")
        bleurt_model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20-D12")
        bleurt_tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D12")
        bleurt_model.eval()
        if torch.cuda.is_available():
            bleurt_model = bleurt_model.cuda()

        for mid in model_ids:
            hyps = hyps_by_model[mid]
            refs_list = [e["source_text"] for e in data]  # English source as "reference" — BLEURT uses src+ref
            try:
                seg_scores = []
                for ref, hyp in zip(refs_list, hyps):
                    inputs = bleurt_tokenizer([ref], [hyp], padding="longest", return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    with torch.no_grad():
                        res = bleurt_model(**inputs).logits.flatten().tolist()
                    seg_scores.append(round(res[0], 4))
                bleurt_scores[mid] = seg_scores
                mean_score = np.mean(seg_scores)
                print(f"  {mid}: BLEURT mean={mean_score:.4f}")
            except Exception as e:
                print(f"  {mid}: BLEURT ERROR — {str(e)[:100]}")
                bleurt_scores[mid] = [None] * len(data)
        del bleurt_model; gc.collect(); torch.cuda.empty_cache()
    except ImportError as e:
        print(f"  BLEURT not available: {e}")
        bleurt_scores = {mid: [None]*len(data) for mid in model_ids}

    # ── 8 & 9. COMET / COMET-Kiwi ───────────────────────────────────────
    print("\n--- COMET / COMET-Kiwi ---")
    comet_scores = {}
    kiwi_scores = {}
    try:
        from benchmark.quality.metrics_comet import compute_comet, compute_comet_kiwi
        for mid in model_ids:
            hyps = hyps_by_model[mid]
            srcs = [e["source_text"] for e in data]
            refs_list = [e["source_text"] for e in data]
            # COMET-22 (reference-based) — note: using source as pseudo-reference
            # In production, use the golden Turkish references
            try:
                comet_result = compute_comet(srcs, hyps, refs_list)
                comet_score = comet_result.get("system_score")
            except Exception as e:
                print(f"  {mid}: COMET ERROR — {str(e)[:100]}")
                comet_score = None
            # COMET-Kiwi (reference-free)
            try:
                kiwi_result = compute_comet_kiwi(srcs, hyps)
                kiwi_score = kiwi_result.get("system_score")
            except Exception as e:
                print(f"  {mid}: COMET-Kiwi ERROR — {str(e)[:100]}")
                kiwi_score = None
            comet_scores[mid] = comet_score
            kiwi_scores[mid] = kiwi_score
            print(f"  {mid}: COMET={comet_score}, Kiwi={kiwi_score}")
    except Exception as e:
        print(f"  COMET module error: {str(e)[:100]}")

    # ── 10. MetricX-24 ──────────────────────────────────────────────────
    print("\n--- MetricX-24 ---")
    metricx_scores = {}
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        torch.cuda.empty_cache(); gc.collect()
        print("  Loading google/metricx-24-hybrid-large-v2p6...")
        mx_tokenizer = AutoTokenizer.from_pretrained("google/metricx-24-hybrid-large-v2p6")
        mx_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/metricx-24-hybrid-large-v2p6",
            torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
        mx_model.eval()

        for mid in model_ids:
            hyps = hyps_by_model[mid]
            srcs = [e["source_text"] for e in data]
            refs_list = [e["source_text"] for e in data]  # pseudo-ref
            try:
                seg_scores = []
                for src, ref, hyp in zip(srcs, refs_list, hyps):
                    # MetricX input: "source: {src} reference: {ref} candidate: {hyp}"
                    input_text = f"source: {src} reference: {ref} candidate: {hyp}"
                    inputs = mx_tokenizer(
                        input_text, return_tensors="pt", truncation=True,
                        max_length=1536, padding=False,
                    ).to("cuda:0")
                    with torch.no_grad():
                        batch_size = inputs["input_ids"].size(0)
                        decoder_input_ids = torch.zeros(
                            (batch_size, 1),
                            dtype=torch.long,
                            device=inputs["input_ids"].device,
                        )
                        outputs = mx_model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            decoder_input_ids=decoder_input_ids,
                            return_dict=True,
                        )
                        lm_logits = outputs.logits  # shape: [batch_size, 1, vocab_size]
                        pred = lm_logits[:, 0, 250089]
                        pred = torch.clamp(pred, 0.0, 25.0)
                    try:
                        score = float(pred.item())
                    except Exception:
                        score = None
                    seg_scores.append(score)
                metricx_scores[mid] = seg_scores
                valid = [s for s in seg_scores if s is not None]
                if valid:
                    print(f"  {mid}: MetricX-24 mean={np.mean(valid):.4f}")
                else:
                    print(f"  {mid}: MetricX-24 — all None (model output unparseable)")
            except Exception as e:
                print(f"  {mid}: MetricX-24 ERROR — {str(e)[:100]}")
                metricx_scores[mid] = [None] * len(data)
        del mx_model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  MetricX-24 not available: {str(e)[:200]}")
        metricx_scores = {mid: [None]*len(data) for mid in model_ids}

    # ── Compile final metrics ───────────────────────────────────────────
    print("\n--- Compiling final metrics ---")
    results = []
    for sid_idx, entry in enumerate(data):
        sid = entry["source_id"]
        src = entry["source_text"]
        for mid in model_ids:
            m = entry["models"][mid]
            idx = model_ids.index(mid) * len(data) + sid_idx
            item = {
                "source_id": sid,
                "source_text": src,
                "model_id": mid,
                "translation": m["text"],
                "output_tokens": m["output_tokens"],
                "latency_ms": m["latency_ms"],
                "metrics": {
                    "chrf": chrf_scores.get(mid) if sid == 0 else None,
                    "chrf_pp": chrfpp_scores.get(mid) if sid == 0 else None,
                    "chrf_seg": all_metrics[idx].get("chrf_seg") if idx < len(all_metrics) else None,
                    "bleu": bleu_scores.get(mid) if sid == 0 else None,
                    "sacrbleu": bleu_scores.get(mid) if sid == 0 else None,
                    "spbleu": spbleu_scores.get(mid) if sid == 0 else None,
                    "morph_bleu": morph_scores.get(mid) if sid == 0 else None,
                    "bleurt": bleurt_scores.get(mid, [None]*len(data))[sid] if mid in bleurt_scores else None,
                    "comet": comet_scores.get(mid),
                    "comet_kiwi": kiwi_scores.get(mid),
                    "metricx_24": metricx_scores.get(mid, [None]*len(data))[sid] if mid in metricx_scores else None,
                },
            }
            results.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} metric entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

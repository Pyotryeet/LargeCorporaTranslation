# Quality Pipeline Assessment — EN→TR Translation Evaluation

**Date:** June 2026  
**Codebase version:** v3.7  
**Author:** Implementation audit + recommendations

> This document audits the current quality evaluation pipeline against academic
> best practice, rates the planned ideas in `QUALITY_METRICS_RESEARCH.md` and
> `LANGUAGE_EVALUATION_STUDY.md`, and proposes concrete new improvements with
> implementation notes.
>
> Read alongside:
> - [`QUALITY_METRICS_RESEARCH.md`](Research/QUALITY_METRICS_RESEARCH.md) — metric landscape research
> - [`LANGUAGE_EVALUATION_STUDY.md`](Research/LANGUAGE_EVALUATION_STUDY.md) — Turkish-specific challenges
> - [`benchmark/quality/`](../benchmark/quality/) — implementation

---

## Part 1 — Current Implementation: What's Actually Running

### The Stack (as wired in `benchmark.py`)

```
benchmark/quality/
├── benchmark.py          ← orchestrator (QualityBenchmark.run)
├── metrics_comet.py      ← COMET-22 (reference-based) + COMET-Kiwi (reference-free, gated)
├── metrics_bertscore.py  ← BERTScore (reference-based, bert-base-multilingual-cased)
├── metrics_bleu.py       ← BLEU (n-gram)
├── metrics_chrf.py       ← chrF++ (character n-gram)
└── references.py         ← single-reference JSONL loader
```

### What runs in parallel

`_compute_metrics_parallel` fires all 5 metrics concurrently via `ThreadPoolExecutor`.
Each has a 600s timeout. The results include per-segment scores and low-quality segment lists.

### Honest gap list (from `benchmark.py` line 12–20)

The code's own docstring already admits the core problems:

| Gap | Severity | Academic consequence |
|-----|----------|----------------------|
| Single reference per source | 🔴 Critical | COMET-22 scores are unreliable; surface-valid translations penalized |
| No bootstrap confidence intervals | 🔴 Critical | System scores are point estimates with no statistical meaning |
| No paired significance testing | 🔴 Critical | Cannot tell if one model is better than another |
| 10-pair golden set (smoke test) | 🟡 High | 10 samples have essentially zero statistical power |
| COMET-Kiwi is gated on HuggingFace | 🟡 High | Falls back silently to `None` if access not granted |
| BERTScore is primary metric | 🟠 Medium | BERTScore was designed for reference-based scoring; lower correlation than COMET for MT |
| Fixed batch size, no progressive sampling | 🟠 Medium | Always evaluates full set — wastes time at good quality, not enough at bad quality |

---

## Part 2 — Rating the Existing Plans

### From `QUALITY_METRICS_RESEARCH.md`

| Idea | My Rating | Verdict |
|------|-----------|---------|
| **xCOMET-lite as Tier 1 metric** | ⭐⭐⭐⭐⭐ | Excellent. 278M params, 154 samples/s, reference-free, fits alongside translation model on H200. Should replace BERTScore as the primary continuous metric. |
| **xCOMET-XL for statistical quality gate** | ⭐⭐⭐⭐ | Very good. The progressive sampling strategy (100→500→1K→2K) is academically sound and saves 95% of evaluation time in the common case. |
| **Paired bootstrap resampling** | ⭐⭐⭐⭐⭐ | This is the WMT standard since 2004. The code snippet in the doc is correct and ready to implement. Should be P0. |
| **Multi-reference set (3 human translators × 500 sentences)** | ⭐⭐⭐⭐ | Right idea. Fixes the root cause of COMET-22 unreliability. ~$5-10K cost. Worth it if publishing results. |
| **Back-translation verification (EN→TR→EN)** | ⭐⭐⭐ | Good sanity check, not a primary metric. NLLB 600M already runs on the benchmark so the roundtrip is essentially free compute. |
| **fastText domain classifier + per-domain scores** | ⭐⭐⭐⭐ | Critical for a 6.23T heterogeneous corpus. Without domain stratification, system-level scores hide severe quality failures in specific domains (legal, medical, social media). |
| **Remove COMET-22 from default path** | ⭐⭐⭐⭐ | Correct. Single-reference COMET-22 scores are actively misleading. Replace with xCOMET reference-free. |
| **Keep chrF++ for legacy comparison** | ⭐⭐⭐ | Reasonable. chrF++ is the best n-gram metric for Turkish (character-level, robust to morphology) but shouldn't gate decisions. |
| **Replace BLEU as a gate** | ⭐⭐⭐⭐⭐ | Correct. BLEU on Turkish is nearly meaningless (one Turkish word = many English words; n-gram matching fails catastrophically). Keep for historical comparison only. |

### From `LANGUAGE_EVALUATION_STUDY.md`

The document proposes the **TTQS framework** (Turkish Translation Quality Scoring):

| TTQS Component | Weight | My Rating | Notes |
|---------------|--------|-----------|-------|
| chrF++ | 20% | ⭐⭐⭐ | Good choice for Turkish morphology, correct weight |
| MetricX-24 | 40% | ⭐⭐⭐⭐ | Strong reference-based metric, but single-reference problem applies here too |
| COMET-Kiwi | 20% | ⭐⭐⭐⭐ | Good. But gated on HuggingFace. Use xCOMET reference-free as drop-in replacement. |
| LLM-as-a-Judge | 20% | ⭐⭐ | Academically sound for system-level ranking, but 44× slower than COMET. At 6.23T scale this is infeasible as a regular metric. Reserve for occasional calibration runs only. |

**Overall TTQS rating: ⭐⭐⭐½** — The composite approach is the right philosophy. The weights need to be empirically calibrated with native Turkish speaker judgments (the doc correctly identifies this). The LLM-as-Judge weight should be zero for continuous evaluation and non-zero only for periodic audits.

### From `NLLB_MADLAD_BENCHMARKS.md §4` (Future opportunities)

| Idea | My Rating | Notes |
|------|-----------|-------|
| FP8 KV-Cache for KV ceiling | ⭐ | **Reverted.** 0% speedup at this model scale. |
| FlashAttention-3 | ⭐⭐⭐ | Real but marginal for seq2seq; already auto-dispatched |
| Fused Triton Decoder Loop | ⭐⭐⭐⭐ | High value — bypasses HF generate() Python overhead |
| Seq2Seq Speculative Decoding | ⭐⭐ | Low value at high batch sizes (batch=512→2048 is already saturated) |

---

## Part 3 — New Ideas

### 🆕 Idea 1: Morpheme-BLEU for Turkish

**Rating: ⭐⭐⭐⭐⭐ (highest value new idea)**  
**Effort:** Medium (2–3 days) | **Risk:** Low | **Dependencies:** `morfessor` or `zeyrek`

**What it is:** Standard BLEU scores Turkish at the word level. One Turkish word can correspond to 3–5 English words — a morphologically valid translation gets a BLEU score of 0 for every suffix that differs from the reference. Morpheme-BLEU segments both hypothesis and reference into morphemes before scoring, aligning with how Turkish actually encodes meaning.

**How:**
```python
import morfessor  # pip install morfessor

def morpheme_tokenize_turkish(text: str, model) -> str:
    """Segment Turkish text into morphemes for BLEU scoring."""
    words = text.split()
    morphemes = []
    for word in words:
        segments, _ = model.viterbi_segment(word)
        morphemes.extend(segments)
    return " ".join(morphemes)

# At evaluation time:
# 1. Load Morfessor model trained on Turkish (available pre-trained)
# 2. Apply morpheme_tokenize_turkish to both hypotheses and references
# 3. Compute standard BLEU on the morpheme-tokenized strings
```

**Why it matters:** The `LANGUAGE_EVALUATION_STUDY.md` explicitly identifies this as "Morpheme-BLEU" and notes it "provides a more accurate measure of morphological correctness." This is a direct, cheap implementation of the key recommendation from the Turkish morphology research.

**Alternative (no new dep):** `zeyrek` is a Python morphological analyzer for Turkish with no extra downloads. Or use character-level n-grams with order ≤ 3 (essentially chrF2) which captures morpheme boundaries without explicit segmentation.

---

### 🆕 Idea 2: Back-Translation Round-Trip Score

**Rating: ⭐⭐⭐⭐**  
**Effort:** Low (1 day) | **Risk:** None | **Dependencies:** None (NLLB already on bench)

**What it is:** Translate EN→TR (hypothesis), then TR→EN with NLLB-600M, then measure semantic similarity between the original English source and the back-translated English. This is **reference-free** and captures adequacy without any Turkish reference needed.

**Why it's more reliable than a single Turkish reference:**
- The back-translation is done in English, where all metrics are well-validated
- It tests whether the *meaning* survived the roundtrip, not whether the surface form matches one specific reference
- NLLB-600M TR→EN is already available in the benchmark (just set `nllb_source_lang="tur_Latn"`, `nllb_target_lang="eng_Latn"`)

**How to score the roundtrip:**
```python
# Use BERTScore between original English source and back-translated English
# (both in English — validated, reliable scoring)
from benchmark.quality.metrics_bertscore import compute_bertscore

roundtrip_score = compute_bertscore(
    references=original_english_sources,
    hypotheses=back_translated_english,
)
# Expected: good translations → roundtrip BERTScore ≥ 0.85
# Hallucinations/wrong language → roundtrip BERTScore < 0.5
```

**What it catches that other metrics miss:**
- Complete hallucinations (Turkish output that has no relation to the source)
- Wrong-language outputs (model outputs English or another language instead of Turkish)
- Meaning-preserving but surface-different translations that a single Turkish reference would penalize

**Where to add it:** `benchmark/quality/metrics_roundtrip.py` — new module, plug into `_compute_metrics_parallel` as a 6th parallel metric. Uses the existing NLLB backend, no new model downloads.

---

### 🆕 Idea 3: Hallucination Detector (Source-Target Length Ratio)

**Rating: ⭐⭐⭐⭐**  
**Effort:** Very low (< 1 day, ~20 lines) | **Risk:** None | **Dependencies:** None

**What it is:** A lightweight rule-based sanity check that flags obviously broken translations before running expensive neural metrics. Turkish translations of English text are typically 10–30% longer in characters (due to agglutination adding suffixes) but should never be 10× shorter or 10× longer.

**Three fast checks:**

```python
def detect_translation_pathologies(
    sources: list[str],
    hypotheses: list[str],
) -> dict:
    """Detect hallucinations and failures without neural metrics."""
    hallucinated = []
    empty_outputs = []
    length_anomalies = []
    wrong_script = []

    for i, (src, hyp) in enumerate(zip(sources, hypotheses)):
        if not hyp or len(hyp.strip()) < 3:
            empty_outputs.append(i)
            continue

        # 1. Length ratio (Turkish is typically 0.8–2.0× English character length)
        ratio = len(hyp) / max(len(src), 1)
        if ratio < 0.15 or ratio > 5.0:
            length_anomalies.append({"idx": i, "ratio": ratio, "src": src[:80], "hyp": hyp[:80]})

        # 2. Script check — Turkish uses Latin alphabet (no Cyrillic, Arabic, etc.)
        non_latin = sum(1 for c in hyp if ord(c) > 0x024F and not c.isspace())
        if non_latin / max(len(hyp), 1) > 0.1:
            wrong_script.append({"idx": i, "hyp": hyp[:80]})

        # 3. Source repetition (model copied source instead of translating)
        overlap = len(set(src.lower().split()) & set(hyp.lower().split()))
        if overlap / max(len(src.split()), 1) > 0.7:
            hallucinated.append({"idx": i, "src": src[:80], "hyp": hyp[:80]})

    total = len(sources)
    return {
        "total": total,
        "empty_rate": len(empty_outputs) / total,
        "hallucination_rate": len(hallucinated) / total,
        "wrong_script_rate": len(wrong_script) / total,
        "length_anomaly_rate": len(length_anomalies) / total,
        "pathological_count": len(empty_outputs) + len(hallucinated) + len(wrong_script),
        "details": {
            "empty": empty_outputs,
            "hallucinated": hallucinated,
            "wrong_script": wrong_script,
            "length_anomalies": length_anomalies,
        }
    }
```

**Why before neural metrics:** If 20% of outputs are empty strings or wrong-script gibberish, running COMET-22 is a waste of GPU time. This catches catastrophic failures in milliseconds.

---

### 🆕 Idea 4: Segment-Level Score Distribution (Not Just System Average)

**Rating: ⭐⭐⭐⭐**  
**Effort:** Low (1 day) | **Risk:** None | **Dependencies:** None (already have segment scores)

**What it is:** The current pipeline computes per-segment COMET/xCOMET scores but only reports the system average. The distribution is far more informative: a system averaging 0.70 could mean "every sentence is decent" OR "half are excellent and half are catastrophically bad."

**Already computed, not reported.** `metrics_comet.py` already returns `segments_scores`. Just need to compute and log the distribution.

**Metrics to add:**
```python
import numpy as np

def score_distribution(scores: list[float]) -> dict:
    arr = np.array(scores)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),   # worst 10%
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),   # best 10%
        "pct_below_0.5": float((arr < 0.5).mean()),  # catastrophic rate
        "pct_below_0.65": float((arr < 0.65).mean()), # marginal rate
        "pct_above_0.75": float((arr > 0.75).mean()), # strong rate
    }
```

**Why this matters for a 6.23T corpus:** A model with 5% catastrophically bad translations (score < 0.5) but 95% good ones will look fine on the system average. At 6.23T tokens, 5% catastrophic = 311 billion tokens of bad output. The distribution catches this; the average hides it.

---

### 🆕 Idea 5: Paired Significance Testing Between Model Checkpoints

**Rating: ⭐⭐⭐⭐⭐ (essential for research validity)**  
**Effort:** Low (1 day) | **Risk:** None | **Dependencies:** `numpy` (already installed)

**What it is:** When comparing two models (NLLB-600M vs NLLB-1.3B, or before/after an optimization), the current pipeline gives two point estimates with no way to know if the difference is statistically real. Bootstrap resampling gives 95% confidence intervals.

**The code from `QUALITY_METRICS_RESEARCH.md §5.1` is already correct and ready to use.** It just needs to be wired into `QualityBenchmark` as a comparison method:

```python
# In benchmark/quality/benchmark.py — add this method to QualityBenchmark:

def compare_to_baseline(
    self,
    baseline_results: QualityResults,
    candidate_results: QualityResults,
    metric: str = "comet_kiwi",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap test: is candidate significantly better than baseline?"""
    import numpy as np

    baseline_segs = baseline_results.comet_kiwi.get("segments_scores", [])
    candidate_segs = candidate_results.comet_kiwi.get("segments_scores", [])

    if len(baseline_segs) != len(candidate_segs) or len(baseline_segs) < 100:
        return {"error": "Need matched pairs (≥100 segments each)"}

    rng = np.random.default_rng(42)
    n = len(baseline_segs)
    obs_diff = np.mean(candidate_segs) - np.mean(baseline_segs)

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs.append(
            np.mean([candidate_segs[i] - baseline_segs[i] for i in idx])
        )

    ci_lower, ci_upper = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "metric": metric,
        "observed_diff": round(float(obs_diff), 4),
        "ci_95": [round(float(ci_lower), 4), round(float(ci_upper), 4)],
        "significantly_better": bool(ci_lower > 0),
        "significantly_worse": bool(ci_upper < 0),
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
    }
```

---

### 🆕 Idea 6: Domain-Stratified Sampling with fastText

**Rating: ⭐⭐⭐⭐**  
**Effort:** Medium (2–3 days) | **Risk:** Low | **Dependencies:** `fasttext` (lightweight, 900KB model)

**What it is:** The 6.23T token corpus spans news, web, legal, medical, and social media domains. A single system-level score aggregates across all domains and hides domain-specific failures. A model might be excellent at news but catastrophic at legal text.

**How:**
1. Download `lid.176.ftz` (fastText language ID model, 900KB, identifies language and domain signals)
2. For EN text, use a pre-trained topic classifier (fastText trained on Wikipedia categories) to assign a domain label per sentence
3. Report quality scores grouped by domain

**Expected domains for EN→TR at scale:**
- `news` — well-represented in NLLB training data; expect high quality
- `web` — mixed; NLLB degrades on colloquial/informal text
- `legal` — specialized terminology; expect moderate quality
- `medical` — specialized; expect lowest quality (medical Turkish is highly technical)
- `social` — informal Turkish; emoticons and slang cause failures

**Where to add it:** New `benchmark/quality/domain_classifier.py`. Plug into `QualityBenchmark.run()` to stratify the sample before computing metrics. Each domain gets its own score distribution.

---

### 🆕 Idea 7: Turkish-Specific Vowel Harmony Validator

**Rating: ⭐⭐⭐ (research contribution)**  
**Effort:** High (3–5 days) | **Risk:** Low | **Dependencies:** `zeyrek` or rule-based

**What it is:** Turkish vowel harmony is a hard phonological constraint — if a suffix violates vowel harmony, the word is ungrammatical. Neural metrics score this as a small difference; human readers notice it immediately.

**How (rule-based, zero-dependency):**
```python
FRONT_VOWELS = set("eiöü")
BACK_VOWELS = set("aıou")

def check_vowel_harmony(word: str) -> bool:
    """Return True if the word's vowels all agree in front/back classification."""
    vowels = [c for c in word.lower() if c in FRONT_VOWELS | BACK_VOWELS]
    if len(vowels) < 2:
        return True  # single-vowel words always pass
    vowel_types = [c in FRONT_VOWELS for c in vowels]
    # All vowels should be the same type (all front or all back)
    # Allow one transition point (compound words)
    transitions = sum(1 for a, b in zip(vowel_types, vowel_types[1:]) if a != b)
    return transitions <= 1  # 0 = perfect harmony, 1 = one compound

def vowel_harmony_score(hypotheses: list[str]) -> dict:
    """Fraction of Turkish words that pass vowel harmony check."""
    total_words = 0
    violations = 0
    for hyp in hypotheses:
        for word in hyp.split():
            # Strip punctuation
            word = word.strip(".,;:!?\"'()-")
            if len(word) > 2 and any(c in FRONT_VOWELS | BACK_VOWELS for c in word.lower()):
                total_words += 1
                if not check_vowel_harmony(word):
                    violations += 1
    rate = violations / max(total_words, 1)
    return {
        "vowel_harmony_violation_rate": round(rate, 4),
        "total_words_checked": total_words,
        "violations": violations,
        # Interpretation: < 0.05 = good, 0.05-0.15 = moderate issues, > 0.15 = systematic problem
    }
```

**Why it matters:** No existing neural metric specifically validates vowel harmony. A model with systematic vowel harmony violations will score 0.70 on COMET-Kiwi (still "acceptable") but produce outputs that native Turkish speakers find obviously wrong. This is a gap that the docs correctly identify as an "open research problem."

---

## Part 4 — Implementation Priority

```
P0 — Essential for any published result (2–3 days total):
  1. Bootstrap confidence intervals on all system scores        (QUALITY_METRICS_RESEARCH.md §5.1)
  2. Paired significance test between model runs                (Idea 5 above)
  3. Hallucination / pathology detector                        (Idea 3 above — run first, cheapest)
  4. Score distribution reporting (p10/p25/p50/p75/p90)        (Idea 4 above — already computed)

P1 — Major quality improvements (1 week):
  5. Replace BERTScore with xCOMET-lite as primary metric       (QUALITY_METRICS_RESEARCH.md §7)
  6. Back-translation round-trip score via NLLB                 (Idea 2 above)
  7. Progressive sampling (100→500→1K→2K) with early exit       (QUALITY_METRICS_RESEARCH.md §5.3)

P2 — Research contributions (2+ weeks):
  8. Domain-stratified scoring via fastText classifier          (Idea 6 above)
  9. Morpheme-BLEU for Turkish                                  (Idea 1 above)
  10. Vowel harmony validator                                   (Idea 7 above)
  11. Multi-reference Turkish golden set (500 sentences × 3 refs) (external cost, ~$5-10K)
```

---

## Part 5 — Files to Create / Modify

| File | Action | Change |
|------|--------|--------|
| [`benchmark/quality/benchmark.py`](../benchmark/quality/benchmark.py) | Modify | Add `compare_to_baseline()`, `score_distribution()`, progressive sampling |
| [`benchmark/quality/metrics_comet.py`](../benchmark/quality/metrics_comet.py) | Modify | Swap `DEFAULT_COMET_MODEL` to `xCOMET-lite` as primary; keep COMET-22 as secondary |
| `benchmark/quality/metrics_xcometlite.py` | **New** | xCOMET-lite wrapper (178M, reference-free, fast) |
| `benchmark/quality/metrics_roundtrip.py` | **New** | Back-translation roundtrip score via existing NLLB backend |
| `benchmark/quality/metrics_pathology.py` | **New** | Hallucination/pathology detector (length ratio, script check, copy detection) |
| `benchmark/quality/domain_classifier.py` | **New** | fastText domain classifier + per-domain score aggregation |
| `benchmark/quality/metrics_morpheme.py` | **New** | Morpheme-BLEU + vowel harmony validator |
| `benchmark/quality/bootstrap.py` | **New** | Paired bootstrap CI + significance test (extract from docs) |
| [`data/references.jsonl`](../data/) | Expand | Grow from 10-pair smoke test to 500–2000 sentence evaluation set |

---

## Summary

The current quality pipeline is **functional but statistically fragile**:
- Point estimates with no confidence intervals mean you cannot publish these numbers
- A 10-pair reference set has no statistical power
- BERTScore is the wrong primary metric for MT (COMET/xCOMET is the field standard)
- The pipeline has no detection for catastrophic failures (hallucinations, wrong language)

The research in `QUALITY_METRICS_RESEARCH.md` correctly identifies all these problems and proposes the right fixes. The TTQS framework in `LANGUAGE_EVALUATION_STUDY.md` is well-reasoned but needs the LLM-as-Judge weight reduced to zero for continuous evaluation.

**The single highest-ROI change:** Add bootstrap confidence intervals and the paired significance test (P0, ~4 hours, zero new dependencies). This immediately makes every quality result in this benchmark publishable.

**The single most impactful metric replacement:** Swap primary metric from BERTScore → xCOMET-lite (P1, ~2 hours). xCOMET-lite is 2× faster, purpose-built for MT, and reference-free — solving the single-reference problem without any data collection.

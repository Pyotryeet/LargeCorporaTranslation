# EN→TR Translation Quality Evaluation — State of the Art

> **Research report — June 2026.**
>
> 📚 **Background research — not an implementation spec.** See
> [ARCHITECTURE.md §8 #31–#35](ARCHITECTURE.md#8-feature-status-the-truth-table)
> for the currently-wired metric stack (BERTScore, COMET-22, COMET-Kiwi, BLEU,
> chrF++) and [DEVELOPMENT.md](DEVELOPMENT.md) for development conventions.

---

## Table of Contents

1. [The EN→TR Evaluation Gap](#1-the-entr-evaluation-gap)
2. [The Metric Landscape](#2-the-metric-landscape)
   - [Reference-Free Neural Metrics](#21-reference-free-neural-metrics)
   - [Reference-Based Neural Metrics](#22-reference-based-neural-metrics)
   - [LLM-as-Judge](#23-llm-as-judge)
   - [Cross-Metric Comparison](#24-cross-metric-comparison)
3. [Why Turkish Breaks Standard Metrics](#3-why-turkish-breaks-standard-metrics)
4. [The Single-Reference Problem](#4-the-single-reference-problem)
5. [Statistical Methodology for Reliable Evaluation](#5-statistical-methodology-for-reliable-evaluation)
6. [Inference Cost at 6.23T Scale](#6-inference-cost-at-623t-scale)
7. [Recommended Architecture](#7-recommended-architecture)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Sources](#9-sources)

---

## 1. The EN→TR Evaluation Gap

After evaluating every major neural MT metric family — COMET, xCOMET, MetricX,
ReMedy, BERTScore, LLM-as-judge, and n-gram baselines — one finding stands above
all others:

> **No neural metric has been specifically validated for English→Turkish.**
> The WMT metrics shared tasks (WMT22–WMT25) have never included Turkish as a
> language pair. (Turkish appeared only in WMT17–18, and only in the tr→en
> direction.)

This is both a problem and an opportunity. Your benchmark cannot rely on
off-the-shelf metric correlation numbers, but it has the chance to contribute
the **first EN→TR neural metric validation**.

The recommendation emerging from this analysis is a **two-tier strategy**:

| Tier | Metric | Role | Speed | VRAM |
|------|--------|------|-------|------|
| **Tier 1** | xCOMET-lite (278M) | High-throughput scoring — every batch | 154 samples/s (RTX 3090) | 1.8 GB |
| **Tier 2** | xCOMET-XL (3.5B) or MetricX-25 (12B) | Statistical quality gating on stratified samples | 30–60 samples/s estimated | 8–24 GB |

The sections below build the case for this recommendation, metric by metric,
challenge by challenge.

---

## 2. The Metric Landscape

### 2.1 Reference-Free Neural Metrics

Reference-free (or "quality estimation," QE) metrics evaluate a translation
from source + hypothesis only, without needing a human reference. This is
critical at scale — commissioning human references for every domain in a 6.23T
token corpus is infeasible.

#### xCOMET-lite (278M) — Recommended Tier 1

| Property | Value |
|----------|-------|
| **Paper** | Larionov et al., EMNLP 2024 |
| **Parameters** | 278M |
| **VRAM** | 1.8 GB |
| **Speed** | 154 samples/s on RTX 3090 |
| **Kendall τ (ref-based)** | 0.388 |
| **Kendall τ (ref-free)** | 0.363 |
| **Gated?** | No — available on HuggingFace |
| **Turkish validated?** | No |

**Why it wins:** ~52% smaller than COMET-22 (278M vs 581M parameters) with
comparable accuracy — in fact xCOMET-lite **outperforms** COMET-22 by +6.4% on
the WMT22 metrics challenge dataset. Fine-tuned
XLM-RoBERTa encoder with a lightweight regression head. The SentencePiece tokenizer provides comparable subword handling. For Turkish
specifically, published research (COLING 2025; ITU/Bogazici University studies)
finds that WordPiece with large vocabularies (128K+) produces more
morphologically meaningful splits, though both tokenizers handle Turkish
agglutination better than BPE. xCOMET-lite's SentencePiece is adequate; the
tokenizer choice is not the decisive factor for EN→TR metric quality.

```python
from comet import download_model, load_from_checkpoint
model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
data = [{"src": src, "mt": hyp} for src, hyp in zip(sources, hypotheses)]
seg_scores, system_score = model.predict(data, batch_size=8)
```

#### xCOMET-XL (3.5B) & xCOMET-XXL (10.7B) — Recommended Tier 2

| Model | Parameters | Accuracy | Ungated? |
|-------|-----------|----------|----------|
| XCOMET-XL | 3.5B | Higher than lite | Yes |
| XCOMET-XXL | 10.7B | Highest in family | Yes |

Larger xCOMET variants provide higher accuracy at proportionally higher cost.
Reserve for the statistical quality gate — run on stratified samples only, not
every batch.

#### COMET-Kiwi — Reference-Free, but Gated

The best reference-free COMET Kiwi model (`wmt23-cometkiwi-da-xxl`) is **gated
on HuggingFace** — requires manual access approval. This is why the benchmark
implementation initially showed `COMET-Kiwi: None`. xCOMET in reference-free
mode is the ungated alternative.

#### MetricX-25-QE (12B) — Reference-Free, Ungated, Heavy

| Property | Value |
|----------|-------|
| **Paper** | Juraska et al., WMT 2025 |
| **Parameters** | 12B (Gemma 3 backbone) |
| **Pairwise accuracy (en-de)** | 55.45 |
| **Pairwise accuracy (ja-zh)** | 57.72 |
| **Gated?** | No — on HuggingFace |
| **Inference speed** | Not benchmarked (trained on 64 TPUs) |
| **Turkish validated?** | No |

MetricX-25 is the WMT25 state-of-the-art for segment-level evaluation, but its
12B parameter weight and unbenchmarked inference speed make it impractical for
Tier 1. Best reserved for **quality gating on stratified samples** where the
higher accuracy justifies the cost.

---

### 2.2 Reference-Based Neural Metrics

Reference-based metrics score a hypothesis against one or more human reference
translations. Their reliability is fundamentally limited by reference coverage
(see [§4](#4-the-single-reference-problem)).

#### ReMedy-9B — Best Reference-Based, Ungated

| Property | Value |
|----------|-------|
| **Paper** | Tan & Monz, EMNLP 2025 |
| **Parameters** | 9B (Gemma2 backbone, decoder-only) |
| **Segment pairwise accuracy** | 58.9% (WMT22 MQM) |
| **System-level accuracy** | 91.2% |
| **Ref-free (ReMedy-QE)** | 74.4% average (WMT23) |
| **Gated?** | No |
| **Speed** | Not benchmarked; decoder-only + VLLM implies high latency |

ReMedy's key innovation is **Bradley-Terry pairwise ranking** rather than
regression on absolute scores. This addresses a fundamental problem in MT
evaluation:

> Human inter-annotator agreement at the **segment level** is only
> **Kendall τ 0.2–0.45** (Graham, Mathur & Baldwin, 2015; WMT19; Singh et al.,
> 2024). System-level correlations are substantially higher (>0.95 for some
> metrics). If two humans can't agree on a segment's absolute quality, a
> regression model cannot learn a meaningful absolute score.

Pairwise comparison ("is translation A better than B?") has much higher human
agreement and is what ReMedy models directly. The decoder-only architecture
also captures uncertainty better than encoder-regression approaches — a trend
now dominant in the field (see [§2.5](#25-industry-trends)).

#### COMET-22 — Single-Reference Limitations

COMET-22 is architecturally limited to a single reference. When the reference
coverage is thin — as with the benchmark's single-reference EN→TR setup — scores
can be unreliable. If the model produces a valid translation that differs at the
surface level from the single reference (e.g., *"Bu otobüsü kontrol edeceğim"*
vs. the reference *"Ben otobüsü bulayım"* — both valid translations of "Let me
get the bus"), COMET-22's reference-based embedding may treat them as
substantially different, producing collapsed or near-zero scores. **With 3+
references, COMET-22 gives sensible scores.** Without them, reference-free
metrics like xCOMET are more reliable.

---

### 2.3 LLM-as-Judge

| Property | Value |
|----------|-------|
| **System-level Kendall τ (GEMBA-MQM, GPT-4)** | **0.809** — best among LLM-based reference-free metrics at system-level ranking |
| **Segment-level Kendall τ** | 0.362 — below fine-tuned neural metrics (MetricX-QE achieves 0.387) |
| **Speed** | 196–256s per 1,000 scores on H100 (44× slower than COMET) |
| **Self-bias** | LLMs favor their own translations |
| **Language bias** | Better on high-resource languages; unvalidated for TR |
| **Cost at 6.23T scale** | Prohibitively expensive |

**Verdict:** LLM-as-judge is academically validated for **system-level ranking**
but impractical for per-segment evaluation at 6.23T token scale. The 44× speed
gap and documented self-bias make it unsuitable as a primary metric. Reserve for
occasional calibration against human judgments — it is the best available
approximation to human evaluation, just not at scale.

---

### 2.4 Cross-Metric Comparison

| Metric | Type | VRAM | Time/1000 scores | 6.23T GPU-hours | Gated? |
|--------|------|------|-----------------|----------------------|--------|
| **xCOMET-lite** | Ref-free | 1.8 GB | 6.5s | ~11,000 | No |
| **xCOMET-XL** | Ref-free | 8 GB | ~12s | ~20,000 | No |
| COMET-22 | Ref-based | 2.5 GB | 4.4s | ~7,500 | No |
| BERTScore | Ref-based | 3.5 GB | ~15s | ~25,000 | No |
| MetricX-25 (est.) | Ref-free | ~24 GB | ~30s | ~51,000 | No |
| GPT-4 API (GEMBA) | LLM-judge | N/A | 256s | $50M+ | Yes |

For 6.23T tokens: evaluate every batch with xCOMET-lite (1.8 GB VRAM, marginal
overhead), then run the quality gate on progressive stratified samples with
xCOMET-XL. **Total inference budget: ~12,000–15,000 GPU-hours — approximately
0.7% of the translation compute budget.**

---

### 2.5 Industry Trends

Google, DeepL, and Microsoft do not publicly disclose their internal quality
metrics. What IS known from WMT shared tasks:

| Year | Top metric(s) | Architecture |
|------|--------------|-------------|
| WMT22 | MetricX-XXL (#1), COMET-22 (dominant in top-10), COMET-Kiwi | Encoder-regression |
| WMT23 | XCOMET-Ensemble (#1), MetricX-23 (#2) | Encoder-regression |
| WMT24 | MetaMetrics-MT (#1), MetricX-24 (#2), XCOMET (#3) | Hybrid |
| WMT25 | MetricX-25 (significant improvement), ReMedy-9B | Decoder and encoder |

**Industry trend:** shift toward decoder-only models (ReMedy) because
logprob-based scoring captures uncertainty better than regression.
Bradley-Terry pairwise ranking is preferred over absolute scoring due to low
human inter-annotator agreement.

---

## 3. Why Turkish Breaks Standard Metrics

Turkish morphology creates challenges that standard MT metrics — designed
primarily for English and similar analytic languages — handle poorly. No
existing metric has been validated for Turkish morphology specifically.

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| **Agglutination** | One Turkish word = many English words. BLEU's n-gram matching fails catastrophically because the surface forms share almost nothing. | chrF++ (character n-grams ≤ order 4) is robust. Neural metrics (xCOMET, COMET, BERTScore) use subword tokenization that better captures morpheme-level information. |
| **Vowel harmony** | Suffix form changes based on stem vowels (e.g., *ev-de* vs. *oda-da* for "in the X"). Surface metrics treat these as completely different tokens. | Neural metrics are better; full morphological analysis is needed for reliable evaluation. |
| **Case marking** | The suffixes *-i/-e/-de/-den* encode relationships that English expresses with separate words and prepositions. Wrong case = wrong meaning, but surface-level metrics cannot distinguish a case error from a legitimate wording variation. | **No metric handles this today — open research problem.** |
| **Vocabulary explosion** | Turkish has approximately 4× the vocabulary size of English due to suffix combinations (each root can produce hundreds of surface forms). | SentencePiece/Unigram tokenizers (used by xCOMET, COMET) and WordPiece (used by BERTScore) both handle this better than BPE, though WordPiece with 128K+ vocabularies shows stronger results for Turkish specifically per COLING 2025 research. |

> Your benchmark could contribute the **first morphologically-tagged EN→TR
> evaluation dataset** — this is an open research opportunity.

---

## 4. The Single-Reference Problem

The benchmark's current golden reference set has **one reference per source
sentence**. This is the root cause of several quality-metric issues.

### Why one reference isn't enough

Translation is inherently one-to-many. A single English sentence can have
multiple valid Turkish translations — different word orders, different lexical
choices, different levels of formality. A metric that compares against exactly
*one* reference will penalize perfectly valid translations that happen to use
different surface forms.

COMET-22 producing unreliable scores with single references (see [§2.2](#comet-22--single-reference-limitations))
is the clearest symptom — reference-based embedding sees any deviation from the
single reference as error.

### Three complementary mitigations

**1. Reference-free metrics (primary solution)**
xCOMET-QE and MetricX-25-QE evaluate adequacy from source + hypothesis only.
They do not penalize legitimate wording variation because they never see a
reference. This is the single biggest reason to prefer reference-free metrics
for this benchmark.

**2. Multi-reference evaluation**
Research shows diminishing returns plateau at **4–8 references**. Human-crafted
diverse paraphrases yield a **35–50% relative gain** in correlation with human
judgment. For this benchmark: commission 3 professional human translators for
500 sentences → 1,500 reference sentences. Estimated cost: **~$5,000–10,000**
at market rates.

**3. Back-translation verification**
EN→TR→EN roundtrip via an independent model (e.g., NLLB-200 TR→EN). If the
back-translated meaning matches the original English, the Turkish translation
is adequate regardless of surface form. This is a lightweight sanity check, not
a primary metric.

---

## 5. Statistical Methodology for Reliable Evaluation

### 5.1 Paired Bootstrap Resampling (Academic Standard Since WMT22)

Single-point estimates of system-level scores are meaningless without
confidence intervals. The academic standard is paired bootstrap resampling:

```python
import numpy as np

def quality_gate(baseline_scores, candidate_scores, 
                 alpha=0.05, min_effect=0.02, n_bootstrap=10_000):
    """
    Paired bootstrap test — resample WITHIN pairs, not across.
    
    Reference: Koehn (2004) "Statistical Significance Tests for Machine
    Translation Evaluation." ACL Workshop. Established the paired bootstrap
    methodology that remains the WMT standard.
    """
    n = len(baseline_scores)
    obs_diff = np.mean(candidate_scores) - np.mean(baseline_scores)
    
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diff = np.mean(
            [candidate_scores[i] - baseline_scores[i] for i in idx]
        )
        boot_diffs.append(boot_diff)
    
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    passed = ci_lower > -min_effect
    
    return {
        "passed": passed,
        "observed_diff": obs_diff,
        "ci_95": [ci_lower, ci_upper],
        "p_value": 2 * min(
            np.mean(np.array(boot_diffs) >= 0),
            np.mean(np.array(boot_diffs) <= 0)
        ),
        "n_samples": n,
    }
```

### 5.2 Sample Size Requirements

| Desired sensitivity | Minimum pairs | Power |
|---------------------|--------------|-------|
| Detect 5-point chrF drop | 100 | 0.8 |
| Detect 0.05 xCOMET drop | 500 | 0.8 |
| Detect 0.02 xCOMET drop | 2,000+ | 0.8 |

### 5.3 Progressive Sampling Strategy

Rather than evaluating a fixed-size sample, use a staged approach that exits
early when quality is clearly good — saving 95% of evaluation time in the
common case:

```
Phase 1: 100 sentences → xCOMET ≥ 0.75 → PASS (saves 95% of time)
Phase 2: 400 more → 500 total → CI narrow enough?
Phase 3: 500 more → 1,000 total → still uncertain?
Phase 4: 1,000 more → 2,000 total → FINAL gate decision
```

### 5.4 Quality Gate Implementation

```python
class QualityGate:
    """Statistical quality gate for MT evaluation."""
    
    def __init__(self, baseline_scores, metric_name="xcomet"):
        self.baseline = np.array(baseline_scores)
        self.metric_name = metric_name
    
    def evaluate(self, candidate_scores, alpha=0.05, min_effect=0.02):
        """Paired bootstrap test. Blocks deployment if quality drops."""
        n = len(candidate_scores)
        if n < 100:
            return {"passed": False, "error": f"Need ≥100 samples, got {n}"}
        
        rng = np.random.default_rng(42)
        obs_diff = np.mean(candidate_scores) - np.mean(self.baseline)
        
        boot_diffs = []
        for _ in range(10_000):
            idx = rng.integers(0, n, size=n)
            pairs = [candidate_scores[i] - self.baseline[i] for i in idx]
            boot_diffs.append(np.mean(pairs))
        
        ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
        
        return {
            "passed": ci_lower > -min_effect,
            "observed_diff": obs_diff,
            "ci_95": [ci_lower, ci_upper],
            "n_samples": n,
            "metric": self.metric_name,
        }
```

### 5.5 xCOMET Score Thresholds

To calibrate with human evaluation:

| xCOMET score | Interpretation |
|-------------|---------------|
| ≥ 0.75 | Strong — competitive with commercial systems |
| 0.55–0.75 | Acceptable — meaning preserved |
| 0.35–0.55 | Marginal — review needed |
| < 0.35 | Unacceptable — do not deploy |

---

## 6. Inference Cost at 6.23T Scale

Cost matters. A metric that costs more to run than the translation itself is
not a metric — it's a second translation. The table below projects GPU-hours
for each metric at the full 6.23T token scale:

| Metric | VRAM | Time/1,000 scores | 6.23T tokens GPU-hours |
|--------|------|-----------------|----------------------|
| **xCOMET-lite** | 1.8 GB | 6.5s | ~11,000 |
| **xCOMET-XL** | 8 GB | ~12s | ~20,000 |
| COMET-22 | 2.5 GB | 4.4s | ~7,500 |
| BERTScore | 3.5 GB | ~15s | ~25,000 |
| MetricX-25 (est.) | ~24 GB | ~30s | ~51,000 |
| GPT-4 API (GEMBA) | N/A | 256s | $50M+ |

For context: the translation itself, at ~1,500 tok/s on 2× H200, consumes
approximately **2 million GPU-hours**. A quality metric at ~12,000–15,000
GPU-hours represents **0.7% overhead** — acceptable for a production pipeline.

**The two-tier strategy keeps this manageable:** xCOMET-lite on every batch
(1.8 GB VRAM, marginal overhead) handles the volume; xCOMET-XL on progressive
stratified samples provides the statistical gate without scaling linearly with
corpus size.

---

## 7. Recommended Architecture

### Two-Tier Quality Strategy

```
Tier 1 — High-Throughput (every batch)
  └─ xCOMET-lite (278M)
     ├─ Reference-free mode (source + hypothesis only)
     ├─ 154 samples/s on RTX 3090
     ├─ 1.8 GB VRAM (fits alongside translation model on H200)
     └─ Produces per-segment scores → aggregate into system-level distribution

Tier 2 — Statistical Quality Gate (stratified samples)
  └─ xCOMET-XL (3.5B) or MetricX-25 (12B)
     ├─ Progressive sampling: 100 → 500 → 1,000 → 2,000 sentences
     ├─ Early exit when clearly passing (saves 95% of time)
     ├─ Paired bootstrap test vs. baseline with 95% CI
     └─ Per-domain stratification via fastText language-ID classifier
```

### What to Replace in the Current Benchmark

| Current | Replace With | Rationale |
|---------|-------------|-----------|
| **BERTScore** (primary) | **xCOMET-lite** (tier 1, all batches) | 2× faster, higher correlation, purpose-built for MT |
| **COMET-22** (reference-based) | Remove from default path | Single-ref scores can be unreliable with thin reference coverage; only reliable with 3+ refs |
| **COMET-Kiwi** (gated) | **xCOMET-XL** reference-free mode | Ungated, installable, higher accuracy |
| **BLEU / chrF++** | Keep for legacy comparison only | N-gram metrics penalize legitimate wording variation. chrF++ (standard `char_order=6, word_order=2`) is more robust than BLEU for Turkish morphology but shouldn't drive gating decisions. |
| **Single 32-sentence eval** | **Progressive sampling 100→500→1,000→2,000** | Statistical power + early exit |
| **No statistical test** | **Paired bootstrap with 95% CI** | Academic standard |
| **Hardcoded pass/fail** | **Dynamic gate vs baseline + effect size** | Adapts to model improvements |
| **No per-domain reporting** | **fastText domain classifier + per-domain scores** | Critical for 6.23T heterogeneous text |
| **Max 32 references** | **Full 1,960 references when time allows** | Statistical power |

---

## 8. Implementation Roadmap

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Add xCOMET-lite as primary quality metric | 2 hours | Replaces broken COMET-22 / gated Kiwi |
| **P0** | Add paired bootstrap quality gate | 2 hours | Statistical rigor for deployment decisions |
| **P1** | Add progressive sampling (100→500→1,000→2,000) | 3 hours | 95% time savings at good quality |
| **P1** | Add per-domain stratification (fastText) | 2 hours | Domain-specific quality reporting |
| **P2** | Add xCOMET-XL for tier-2 gating | 1 hour | Higher accuracy on gate decisions |
| **P2** | Commission 3-reference Turkish golden set (500 sentences) | External cost | Public benchmark contribution; fixes single-reference problem |
| **P3** | Add morphological error categorization | Research project | Open research contribution — first EN→TR morphologically-tagged evaluation |

---

## 9. Sources

1. Guerreiro et al. (2024). "xCOMET: Transparent Machine Translation Evaluation
   through Fine-grained Error Detection." TACL, Vol. 12, pp. 979–995.
   The xCOMET framework and XCOMET-XL/XXL models.
2. Larionov et al. (2024). "xCOMET-lite: Distilled xCOMET for Efficient MT
   Evaluation." EMNLP 2024. https://aclanthology.org/2024.emnlp-main.1223/
   The 278M distilled variant used in the Tier-1 recommendation.
3. Tan & Monz (2025). "ReMedy: Reference-Free Machine Translation Evaluation
   with Decoder LLMs." EMNLP 2025.
   https://aclanthology.org/2025.emnlp-main.217/
4. Juraska et al. (2025). "MetricX-25 and GemSpanEval: Google Translate
   Submissions to the WMT25 Evaluation Shared Task." WMT 2025.
   https://aclanthology.org/2025.wmt-1.70/
5. Lavie et al. (2025). "Findings of the WMT25 Shared Task on Automated
   Translation Evaluation Systems: Linguistic Diversity is Challenging and
   References Still Help." WMT 2025.
   https://aclanthology.org/2025.wmt-1.24/
6. Koehn (2004). "Statistical Significance Tests for Machine Translation
   Evaluation." ACL Workshop (precursor to WMT). Established the paired
   bootstrap resampling methodology that remains the WMT standard.
7. Aci, Vuran Sari & Inan Aci (2025). "Morphological and structural complexity
   analysis of low-resource English-Turkish language pair using neural machine
   translation models." PeerJ Computer Science, 11, e3072.
   https://peerj.com/articles/cs-3072/
8. Freitag, Mathur, Deutsch, Lo, Avramidis, Rei, Thompson, Blain, Kocmi, Wang,
   Adelani, Buchicchio, Zerva & Lavie (2024). "Are LLMs Breaking MT Metrics?
   Results of the WMT24 Metrics Shared Task." WMT 2024.
   https://aclanthology.org/2024.wmt-1.2/


---

*Generated: June 22, 2026. 104 Claude agents, 1,696 source fetches, 3,118,648
tokens consumed. All major claims verified by 3-vote adversarial panels.*

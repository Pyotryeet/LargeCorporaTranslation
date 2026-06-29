# Model Evaluation Pipeline — EN→TR Translation Quality

## Overview

The evaluation pipeline selects the best English→Turkish translation model from
5 candidates using a combination of automated metrics (7 scores per model) and
blinded human evaluation (1–10 scale). Human ratings are used to learn optimal
weights for each automated metric, producing a single composite quality score
that best correlates with human judgment.

## Pipeline Architecture

```
 Step 1: Sentence Selection
   select_sentences.py
   └→ 30 diverse English sentences from FineWeb (BERT embeddings + k-means)

 Step 2: Translation
   translate_sentences.py
   └→ 5 models translate the same 30 sentences on MPS

 Step 3: Reference Generation
   generate_references.py
   └→ Gemini 3.1 Pro generates golden Turkish references

 Step 4: Metrics Computation
   run_metrics.py
   └→ 7 automated metrics per model + blinded human eval data

 Step 5: Human Evaluation
   index.html (web UI)
   └→ Blinded A/B/C/D labels, 1–10 scale per translation

 Step 6: Weight Tuning
   tune_weights.py
   └→ Linear regression: human ratings → metric weights → final ranking
```

---

## Step 1: Sentence Selection (`select_sentences.py`)

**What it does:** Loads 5,000 documents from the FineWeb English sample, extracts
English sentences (40–300 chars), embeds them with `bert-base-multilingual-cased`,
clusters into 30 groups via k-means, and selects the centroid sentence from each
cluster.

**Output:** `data/output/model_selection/source_sentences.json` — 30 sentences,
113–293 chars, mean 204 chars.

---

## Step 2: Translation (`translate_sentences.py`)

### Models Translated

| Model ID | Architecture | Parameters | HF Path |
|----------|-------------|-----------|---------|
| `nllb_600m` | Encoder-Decoder (NLLB) | 600M | `facebook/nllb-200-distilled-600M` |
| `nllb_1.3b` | Encoder-Decoder (NLLB) | 1.3B | `facebook/nllb-200-distilled-1.3B` |
| `nllb_3.3b` | Encoder-Decoder (NLLB) | 3.3B | `facebook/nllb-200-3.3B` |
| `nllb_moe_54b` | Sparse MoE Encoder-Decoder | 54.4B | `facebook/nllb-moe-54b` |
| `madlad_3b` | T5 Encoder-Decoder | 3B | `google/madlad400-3b-mt` |
| `madlad_10b` | T5 Encoder-Decoder | 10B | `google/madlad400-10b-mt` |
| `translategemma_4b` | Autoregressive (Gemma 3) | 4B | `google/translategemma-4b-it` |
| `translategemma_12b` | Autoregressive (Gemma 3) | 12B | `google/translategemma-12b-it` |
| `translategemma_27b` | Autoregressive (Gemma 3) | 27B | `google/translategemma-27b-it` |

### Issues Encountered and Resolved

**Issue 1: NLLB empty or 1-token output on 60% of sentences**

*Root cause:* `mask_length=256` was too tight for source text (max 86 tokens)
plus forced BOS, leaving zero decoder budget. `src_lang='eng_Latn'` was being
set on the tokenizer instance AFTER construction by assigning `tok.src_lang`
which silently fails on some HF versions.

*Fix:* (a) Moved `src_lang='eng_Latn'` into the `AutoTokenizer.from_pretrained()`
constructor. (b) Changed `max_new_tokens=256` to `max_length=512`. (c) Added
`num_beams=4` for beam search quality. (d) Used `batch_decode()` as recommended
by HuggingFace docs.

*Source:* Web search — HuggingFace NLLB-200 translation examples, community
discussions on `lang_code_to_id` vs `convert_tokens_to_ids`.

**Issue 2: TranslateGemma 4B echoing English input indefinitely**

*Root cause:* The plain prompt `"Translate English to Turkish:\n{text}"` was
not using the Gemma chat template. Gemma 4B was instruction-tuned with
`<start_of_turn>user` / `<start_of_turn>model` markers. Without proper chat
format and `eos_token_id=[tok.eos_token_id, 106]` (106 = `<end_of_turn>`),
the model generates until `max_new_tokens=256` is exhausted.

*Fix:* (a) Changed to official Gemma chat template:
```
<start_of_turn>user\nTranslate to Turkish:\n{text}<end_of_turn>\n<start_of_turn>model\n
```
(b) Added `eos_token_id=[tok.eos_token_id, 106]` for proper stopping.

**Issue 3: MADLAD output was garbled/empty or repeated indefinitely (e.g., "100000000...")**

*Root cause:* Although MADLAD-400 is a T5-based model, the configuration specifies `"tie_word_embeddings": false` by default. However, the checkpoint does not contain separate parameters for `shared.weight` and `encoder.embed_tokens.weight` under their standard T5 names. Because `tie_word_embeddings` is False, the encoder's embedding layer gets initialized with random weights, causing a massive scale mismatch between the encoder embeddings (scale ~1.9) and the decoder embeddings (scale ~250). This scale mismatch breaks cross-attention, causing the model to generate repetitive gibberish. Setting `decoder_start_token_id=1` (`<s>`) previously was a hack that only prevented loops by forcing malformed punctuation sequences (e.g., `...))) ()`), but didn't fix the underlying quality issue.

*Fix:* 
(a) Manually tie the embedding weights in Python immediately after loading:
```python
m.shared.weight = m.decoder.embed_tokens.weight
m.encoder.embed_tokens.weight = m.decoder.embed_tokens.weight
```
(b) Removed the custom `decoder_start_token_id=1` override, letting it correctly default to `0` (`<unk>`), which is the model's actual trained start token.
(c) Switched from beam search (`num_beams=4`) to greedy decoding (`num_beams=1` by omitting `num_beams` and removing `early_stopping=True`). Because MADLAD-400 is a raw pre-trained model (not heavily instruction-tuned), running beam search on web-crawled inputs can trigger "memorization override" where the model prefers completing a memorized English training sequence over translating it. Greedy decoding forces a Turkish token first, breaking the memorization loop. Kept `no_repeat_ngram_size=3` to protect against generic repetition loops.

*Source:* Deep codebase debugging (inspecting state dict keys and layer weight statistics) and Hugging Face configuration audits.

**Issue 4: NLLB truncation on multi-line documents & HTML entities**

*Root cause:* NLLB is an encoder-decoder model trained primarily on single-sentence parallel Corpora. When fed source texts containing newline characters (`\n`), the encoder's positional representations confuse the decoder, causing it to emit the end-of-sequence token `</s>` prematurely (truncating the rest of the text). Additionally, the NLLB tokenizer represents special characters like ampersands and apostrophes as HTML entities (e.g. `&amp;` and `&apos;` with spacing).

*Fix:* (a) In the NLLB translation loop, split input text on newlines, translate each segment individually, and re-join them with `\n` to perfectly preserve structural layouts. (b) Run `html.unescape()` on NLLB decoded strings to resolve XML/HTML formatting bugs.

**Issue 5: TranslateGemma 4B formatting and surrounding quotes**

*Root cause:* TranslateGemma is an instruction-tuned model. Under loose prompting (e.g., "Translate to Turkish:"), it would occasionally wrap its output in surrounding quote marks, output preambles, or generate markdown bold tags (e.g. `**...**`) copied from source layout configurations.

*Fix:* (a) Strengthened the instruction prompt to explicitly forbid markdown, notes, and surrounding quotes. (b) Added a post-generation processor to strip redundant outer quotes.

**Issue 6: COMET incompatibility with transformers 5.x**

*Root cause:* Supporting TranslateGemma 4B (`gemma3` architecture) requires upgrading to `transformers>=4.48.0` (installed version `5.12.1`). However, COMET `2.2.7` relies on legacy `transformers` API internals:
1. It calls `build_inputs_with_special_tokens` directly on tokenizers, which is deprecated/removed in 5.x.
2. It unpacks a 3-tuple from the model call, which fails in 5.x under `return_dict=False` due to dataclass return changes.

*Fix:* In `run_metrics.py` and `evaluate.py`, added a dynamic monkey patch:
1. Re-implemented `build_inputs_with_special_tokens` on `XLMRobertaTokenizer` classes to delegate to single/pair sequence builders.
2. Monkey-patched `comet.encoders.xlmr.XLMREncoder.forward` to run with `return_dict=True` and access features directly by attribute name (`last_hidden_state`, `hidden_states`) instead of tuple indices, resolving `ValueError: not enough values to unpack (expected 3, got 2)` robustly.

### Final Generation Parameters Per Model

| Parameter | NLLB | MADLAD | TranslateGemma |
|-----------|------|--------|---------------|
| `max_length` | 512 | — | — |
| `max_new_tokens` | — | 200 | 256 |
| `num_beams` | 4 | — (1) | — |
| `do_sample` | — | — | False |
| `decoder_start_token_id` | — | 0 (default) | — |
| `forced_bos_token_id` | `tur_Latn` | — | — |
| `eos_token_id` | — | — | `[eos, 106]` |
| `no_repeat_ngram_size` | — | 3 | — |
| `early_stopping` | — | — | — |

---

## Step 3: Reference Generation (`generate_references.py`)

### Design Decision

We initially used TranslateGemma 4B's output as pseudo-references, but that
circularly biases the evaluation — the model with the best COMET-Kiwi score
automatically scores 100 on every reference-based metric because it IS the
reference.

### Gemini 3.1 Pro Configuration

- **Model:** `gemini-3.1-pro-preview` (latest frontier model as of June 2026)
- **SDK:** `google-genai` v1.65.0 (official Python SDK; `google-generativeai`
  was end-of-lifed November 2025)
- **System instruction:** Expert translator persona with Turkish grammar and
  morphology constraints
- **Temperature:** 0.0 (deterministic)
- **Max output tokens:** 2048

### Issues Encountered and Resolved

**Issue 5: SSL certificate verification failure on macOS Python**

*Root cause:* macOS Python 3.11 from python.org does not include root CA
certificates. `urllib.request.urlopen()` fails with
`CERTIFICATE_VERIFY_FAILED` on the first HTTPS request.

*Fix:* Switched from raw `urllib.request` to the official `google-genai` SDK
which handles TLS internally.

**Issue 6: `thinking_level` parameter config in `google-genai` SDK**

*Root cause:* The `google-genai` SDK structures thinking configuration under a dedicated nested object `thinking_config=types.ThinkingConfig()`, supporting a `thinking_level` attribute (e.g., `"high"`). Passing it as a raw generation config parameter is rejected.

*Fix:* Configure `thinking_level` nested inside `thinking_config`:
```python
config=types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="high"),
)
```

**Issue 7: Initial fallback to gemini-2.5-flash and prompt limitations**

*Root cause:* A fallback to `gemini-2.5-flash` was previously used if the output size was less than 30% of the source text to prevent truncation, but this limited overall quality. Additionally, standard prompts were vulnerable to leaking chain-of-thought steps into the output when reasoning was enabled.

*Fix:* (a) Removed the `gemini-2.5-flash` fallback logic completely, ensuring only the most capable model (`gemini-3.1-pro-preview`) is used. (b) Strengthened the system prompt with strict output formatting rules:
```
You are an expert, professional English-to-Turkish translator with deep knowledge of Turkish grammar, idioms, agglutinative morphology, and natural syntax. Your task is to translate the source text into natural, fluent, and highly accurate Turkish. Fidelity is paramount: match the exact meaning, detail, tone, and formatting of the source. Use natural Turkish structures and vocabulary suited to the context. Strictly output ONLY the translation itself. Do not include any notes, explanations, thinking steps, or introductory/concluding text in the final output.
```

---

## Step 4: Automated Metrics (`run_metrics.py`)

### Metrics Computed

| # | Metric | Type | Scale | Interpretation |
|---|--------|------|-------|---------------|
| 1 | **COMET-Kiwi** | Reference-free neural | 0–1 | Higher = better |
| 2 | **COMET-22** | Reference-based neural | 0–1 | Higher = better |
| 3 | **MetricX-24** | Reference-based neural MQM | 0–25 | **Lower** = better (0 = perfect) |
| 4 | **BERTScore** | Reference-based neural (BERT) | 0–1 | Higher = better |
| 5 | **chrF++** | Character+word n-gram | 0–100 | Higher = better |
| 6 | **spBLEU** | SentencePiece-tokenized BLEU | 0–100 | Higher = better |
| 7 | **Morph-BLEU** | Turkish suffix-stripped BLEU | 0–100 | Higher = better |

### Issues Encountered and Resolved

**Issue 8: MetricX-24 returned `None` — cache corruption**

*Root cause:* The MetricX-24 SentencePiece model file
(`spiece.model`) was partially downloaded to
`~/.cache/huggingface/hub/models--google--metricx-24-hybrid-large-v2p6/.no_exist/`.
Subsequent `AutoTokenizer.from_pretrained()` calls failed with
`TypeError: not a string` from the SentencePiece C++ loader.

*Fix:* Cleared the corrupted HuggingFace cache directory. The model and
tokenizer re-downloaded correctly on the next run.

**Issue 9: MetricX-24 returned `None` — wrong tokenizer source**

*Root cause:* Our code loaded the tokenizer from the MetricX checkpoint
(`google/metricx-24-hybrid-large-v2p6`) which has a fragile SentencePiece
file. The official Google MetricX pipeline uses `google/mt5-xl` as the
tokenizer — a completely separate, stable model with a clean SentencePiece
file. The MetricX model was trained to work with the mT5 tokenizer's
vocabulary.

*Fix:* Changed the tokenizer source to `google/mt5-xl` with a fallback
to the checkpoint tokenizer if unavailable.

*Source:* Web search — Google's `google-research/metricx` GitHub repository
and the `predict.py` CLI documentation which passes `--tokenizer google/mt5-xl`.

**Issue 10: MetricX-24 returned `None` — wrong field order**

*Root cause:* Our code constructed the input as
`source: {src} reference: {ref} candidate: {hyp}`. The official format is
`source: {src} candidate: {hyp} reference: {ref}` — the candidate comes
BEFORE the reference. The model was trained with this specific order;
swapping them produces unparseable output.

*Fix:* Corrected the field ordering to match the official format.

*Source:* The `metricx/predict.py` source code on GitHub shows:
```python
text = f"source: {source} candidate: {hypothesis} reference: {reference}"
```

**Issue 11: BERTScore returned `None`**

*Root cause:* The `bert-score` Python package was never installed in the
virtual environment.

*Fix:* `pip install bert-score`.

**Issue 12: MetricX-24 scores returned `None` (and `null` in metrics.json) during evaluation**

*Root cause:* The model `google/metricx-24-hybrid-large-v2p6` uses the `MT5ForRegression` architecture, which isn't natively supported in Hugging Face `transformers` and falls back to `MT5ForConditionalGeneration`. Standard generation (`model.generate()`) produced textual sentinel tokens (e.g. `<extra_id_0>`) instead of float scores, raising `ValueError` and yielding `None`.

*Fix:* Replaced text generation with a direct model forward pass using a zeroed-out single-step `decoder_input_ids` tensor, and extracted the quality score directly from the logit of the `<extra_id_10>` token (vocab index `250089`) at the first decoder index (clamped to `[0.0, 25.0]`).

**Issue 13: Segment-level chrF++ metrics (`chrf_seg`) scrambled across models and segments**

*Root cause:* `evaluate.py` appended segment-level scores to `all_metrics` grouped by model first, then by segment. However, the compilation step used the formula `idx = sid * len(model_ids) + model_ids.index(mid)`. This mismatched indexing scrambled all segment metrics.

*Fix:* Corrected the offset formula to `idx = model_ids.index(mid) * len(data) + sid_idx` using `enumerate(data)`.

**Issue 14: `PROJECT_ROOT` path resolution errors (`FileNotFoundError` on data paths)**

*Root cause:* In `build_data.py`, `select_sentences.py`, `tune_weights.py`, and `evaluate.py`, `PROJECT_ROOT` resolved via `.parent.parent`, pointing to the `scripts/` folder instead of the workspace root (`H200Research/`).

*Fix:* Changed path resolution to `Path(__file__).resolve().parent.parent.parent` (three levels up) to correctly reference the workspace root.

**Issue 15: `build_data.py` crashed on system-level `metrics.json`**

*Root cause:* Running `build_data.py` on system-level metrics generated by `run_metrics.py` threw `KeyError: 'source_id'` because those metrics are per-model and lack a `source_id` field.

*Fix:* Implemented a robust check using `.get()` to safely construct lookups only when both `source_id` and `model_id` are available.

**Issue 16: `index.html` failed to auto-load `human_eval_data.json`**

*Root cause:* The fetch path was hardcoded to `data/output/...` relative to the current directory, resolving to `scripts/evaluation/data/` (non-existent).

*Fix:* Changed path to `../../data/output/model_selection/human_eval_data.json`.

---

## Step 5: Human Evaluation (`index.html`)

The web UI loads `human_eval_data.json` which contains:
- 30 source sentences (English)
- 4 blinded model translations per sentence (labeled A/B/C/D in random order)
- The mapping from labels to real model IDs is stored in `label_map` but NOT
  displayed — the evaluator cannot know which model produced which translation

Each translation is rated on a 1–10 scale. After all sentences are evaluated,
the user clicks "Download" to save `benchmark_sonuclari.json`.

---

## Step 6: Weight Tuning (`tune_weights.py`)

### Algorithm

Performs linear regression with the human ratings as the dependent variable
and the 7 automated metrics as independent variables. The learned coefficients
are the optimal weights for each metric.

### Input Files

| File | Content |
|------|---------|
| `human_eval_data.json` | Source sentences + label→model mapping |
| `metrics.json` | Per-model per-sentence automated metric scores |
| `benchmark_sonuclari.json` | Human ratings (downloaded from web UI) |

### Output Files

| File | Content |
|------|---------|
| `weights.json` | Learned metric weights + R² score |
| `final_ranking.json` | Model ranking by weighted composite score |

---

## Files Involved

```
scripts/evaluation/
├── select_sentences.py      # Step 1: Sentence selection
├── translate_sentences.py   # Step 2: Translation with 5 models
├── generate_references.py   # Step 3: Gemini 3.1 Pro reference generation
├── run_metrics.py           # Step 4: 7 automated metrics
├── index.html               # Step 5: Web UI for human evaluation
├── tune_weights.py          # Step 6: Weight learning + ranking
├── build_data.py            # (legacy) Human eval dataset builder
└── evaluate.py              # (legacy) Original 10-metric evaluation script

data/output/model_selection/
├── source_sentences.json    # 30 selected English sentences
├── translations.json        # 5 models × 30 translations
├── references.jsonl         # Gemini 3.1 Pro golden references
├── metrics.json             # 7 metrics × 5 models
└── human_eval_data.json     # Blinded data for web UI
```

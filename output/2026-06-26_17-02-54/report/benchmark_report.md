# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T17:05:31Z

## Configuration

- **Backend**: auto
- **Model**: facebook/nllb-200-distilled-600M
- **Target Duration**: 300 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 25008.5 |
| Median tokens/second | 25094.7 |
| P5 tokens/second | 24702.5 |
| P95 tokens/second | 25107.2 |
| Std dev | 210.9 |
| Total output tokens | 3,151,872 |
| Total batches | 6 |

## GPU Utilisation

| Metric | Value |
|---|---|
| Mean GPU utilisation | N/A % |
| P99 GPU utilisation | N/A % |
| Data starvation (<20%) | N/A % |
| Mean GPU temperature | N/A °C |

## Quality Scores

| Metric | Score | Target | Status |
|---|---|---|---|
| BLEU | N/A | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | N/A | >= 0.72 | — |
| BERTScore | N/A | >= 0.55 | — |
| COMET-Kiwi | N/A | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 2873.4 days
- **Bootstrap 95% CI**: [2872.6, 2903.5] days (6 batches)
- **Parametric 95% CI**: [2847.9, 2898.8] days
- **GPU hours**: 137922.0
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
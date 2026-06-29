# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-25T15:00:09Z

## Configuration

- **Backend**: auto
- **Model**: google/translategemma-4b-it
- **Target Duration**: 7200 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 135.6 |
| Median tokens/second | 135.6 |
| P5 tokens/second | 135.6 |
| P95 tokens/second | 135.6 |
| Std dev | 0.0 |
| Total output tokens | 12,815 |
| Total batches | 1 |

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
| BLEU | 0.0 | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | 0.4467 | >= 0.72 | — |
| BERTScore | 0.1987 | >= 0.55 | — |
| COMET-Kiwi | 0.3803 | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 531758.7 days
- **Parametric 95% CI**: [531758.7, 531758.7] days
- **GPU hours**: 12762209.1
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
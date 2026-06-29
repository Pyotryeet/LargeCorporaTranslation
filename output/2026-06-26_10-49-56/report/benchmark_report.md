# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T11:00:34Z

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
| Mean tokens/second | 2723.0 |
| Median tokens/second | 2772.2 |
| P5 tokens/second | 2582.5 |
| P95 tokens/second | 2818.5 |
| Std dev | 95.5 |
| Total output tokens | 758,423 |
| Total batches | 9 |

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
| COMET-22 | 0.7465 | >= 0.72 | ✅ |
| BERTScore | 0.7090 | >= 0.55 | ✅ |
| COMET-Kiwi | 0.7462 | >= 0.72 | ✅ |

## Extrapolation

- **Point estimate**: 26010.6 days
- **Bootstrap 95% CI**: [25959.8, 27098.4] days (9 batches)
- **Parametric 95% CI**: [25309.4, 26711.8] days
- **GPU hours**: 1248507.0
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
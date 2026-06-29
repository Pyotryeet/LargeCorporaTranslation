# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T16:17:12Z

## Configuration

- **Backend**: auto
- **Model**: google/translategemma-4b-it
- **Target Duration**: 300 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 1321.9 |
| Median tokens/second | 1189.1 |
| P5 tokens/second | 925.8 |
| P95 tokens/second | 2134.3 |
| Std dev | 460.0 |
| Total output tokens | 163,410 |
| Total batches | 10 |

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
| BLEU | 0.8 | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | 0.1819 | >= 0.72 | — |
| BERTScore | 0.4744 | >= 0.55 | — |
| COMET-Kiwi | 0.2666 | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 60639.5 days
- **Bootstrap 95% CI**: [44439.4, 65674.3] days (10 batches)
- **Parametric 95% CI**: [45544.4, 75734.7] days
- **GPU hours**: 2910698.1
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
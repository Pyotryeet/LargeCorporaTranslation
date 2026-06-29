# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T13:57:10Z

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
| Mean tokens/second | 483.1 |
| Median tokens/second | 483.1 |
| P5 tokens/second | 483.1 |
| P95 tokens/second | 483.1 |
| Std dev | 0.0 |
| Total output tokens | 62,282 |
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
| BLEU | 0.8 | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | 0.1819 | >= 0.72 | — |
| BERTScore | 0.4744 | >= 0.55 | — |
| COMET-Kiwi | 0.2666 | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 149257.9 days
- **Parametric 95% CI**: [149257.9, 149257.9] days
- **GPU hours**: 7164378.2
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
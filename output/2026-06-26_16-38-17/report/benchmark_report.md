# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T16:40:46Z

## Configuration

- **Backend**: auto
- **Model**: google/translategemma-4b-it
- **Target Duration**: 120 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 3674.8 |
| Median tokens/second | 3660.7 |
| P5 tokens/second | 3614.3 |
| P95 tokens/second | 3755.2 |
| Std dev | 69.2 |
| Total output tokens | 479,755 |
| Total batches | 4 |

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

- **Point estimate**: 19697.5 days
- **Parametric 95% CI**: [19107.2, 20287.7] days
- **GPU hours**: 945477.9
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-25T12:45:55Z

## Configuration

- **Backend**: auto
- **Model**: google/translategemma-4b-it
- **Target Duration**: 7200 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.6.0+cu124

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 1084.7 |
| Median tokens/second | 1115.5 |
| P5 tokens/second | 928.9 |
| P95 tokens/second | 1177.5 |
| Std dev | 106.5 |
| Total output tokens | 73,000 |
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
| BLEU | 0.0 | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | 0.4496 | >= 0.72 | — |
| BERTScore | 0.1785 | >= 0.55 | — |
| COMET-Kiwi | 0.3821 | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 64640.5 days
- **Bootstrap 95% CI**: [62689.5, 72167.8] days (6 batches)
- **Parametric 95% CI**: [57980.1, 71300.9] days
- **GPU hours**: 1551372.1
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
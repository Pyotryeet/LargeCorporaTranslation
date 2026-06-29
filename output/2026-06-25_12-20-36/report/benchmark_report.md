# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-25T12:22:01Z

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
| Mean tokens/second | 1077.9 |
| Median tokens/second | 1108.3 |
| P5 tokens/second | 923.2 |
| P95 tokens/second | 1170.5 |
| Std dev | 105.8 |
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

- **Point estimate**: 65060.4 days
- **Bootstrap 95% CI**: [63084.4, 72613.6] days (6 batches)
- **Parametric 95% CI**: [58358.8, 71762.1] days
- **GPU hours**: 1561450.5
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-25T11:59:31Z

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
| Mean tokens/second | 752.1 |
| Median tokens/second | 744.9 |
| P5 tokens/second | 575.3 |
| P95 tokens/second | 910.6 |
| Std dev | 146.0 |
| Total output tokens | 49,892 |
| Total batches | 5 |

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

- **Point estimate**: 96800.2 days
- **Bootstrap 95% CI**: [84073.5, 113129.5] days (5 batches)
- **Parametric 95% CI**: [73467.9, 120132.5] days
- **GPU hours**: 2323205.2
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
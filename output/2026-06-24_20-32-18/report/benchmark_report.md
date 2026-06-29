# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-24T20:36:14Z

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
| Mean tokens/second | 1528.6 |
| Median tokens/second | 1528.6 |
| P5 tokens/second | 1528.6 |
| P95 tokens/second | 1528.6 |
| Std dev | 0.0 |
| Total output tokens | 87,905 |
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
| COMET-22 | 0.4372 | >= 0.72 | — |
| BERTScore | 0.4076 | >= 0.55 | — |
| COMET-Kiwi | 0.4135 | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 47171.6 days
- **Parametric 95% CI**: [47171.6, 47171.6] days
- **GPU hours**: 2264236.0
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
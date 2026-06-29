# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-25T15:02:10Z

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
| Mean tokens/second | 1445.3 |
| Median tokens/second | 1438.8 |
| P5 tokens/second | 1233.4 |
| P95 tokens/second | 1618.8 |
| Std dev | 149.1 |
| Total output tokens | 98,512 |
| Total batches | 8 |

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
| COMET-22 | 0.4526 | >= 0.72 | — |
| BERTScore | 0.1400 | >= 0.55 | — |
| COMET-Kiwi | 0.3882 | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 50115.7 days
- **Bootstrap 95% CI**: [46912.3, 53666.6] days (8 batches)
- **Parametric 95% CI**: [45793.5, 54438.0] days
- **GPU hours**: 1202777.0
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
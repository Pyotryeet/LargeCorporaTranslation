# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T17:31:20Z

## Configuration

- **Backend**: auto
- **Model**: google/madlad400-3b-mt
- **Target Duration**: 300 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 5476.1 |
| Median tokens/second | 5612.0 |
| P5 tokens/second | 5066.9 |
| P95 tokens/second | 5613.1 |
| Std dev | 304.9 |
| Total output tokens | 656,640 |
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
| BLEU | N/A | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | N/A | >= 0.72 | — |
| BERTScore | N/A | >= 0.55 | — |
| COMET-Kiwi | N/A | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 12848.6 days
- **Bootstrap 95% CI**: [12846.9, 13857.7] days (5 batches)
- **Parametric 95% CI**: [11960.4, 13736.9] days
- **GPU hours**: 616734.0
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-25T12:02:22Z

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
| Mean tokens/second | 733.6 |
| Median tokens/second | 746.5 |
| P5 tokens/second | 576.6 |
| P95 tokens/second | 872.5 |
| Std dev | 142.4 |
| Total output tokens | 38,678 |
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
| BLEU | 0.0 | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | 0.4496 | >= 0.72 | — |
| BERTScore | 0.1785 | >= 0.55 | — |
| COMET-Kiwi | 0.3821 | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 96592.7 days
- **Parametric 95% CI**: [66757.7, 126427.8] days
- **GPU hours**: 2318225.8
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
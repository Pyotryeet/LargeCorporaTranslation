# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T18:09:14Z

## Configuration

- **Backend**: auto
- **Model**: facebook/nllb-200-distilled-600M
- **Target Duration**: 300 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 18018.8 |
| Median tokens/second | 17980.8 |
| P5 tokens/second | 14682.3 |
| P95 tokens/second | 21408.3 |
| Std dev | 3850.3 |
| Total output tokens | 525,312 |
| Total batches | 4 |

## GPU Utilisation

| Metric | Value |
|---|---|
| Mean GPU utilisation | 47.7 % |
| P99 GPU utilisation | 100.0 % |
| Data starvation (<20%) | 51.6 % |
| Mean GPU temperature | 41.8 °C |

## Quality Scores

| Metric | Score | Target | Status |
|---|---|---|---|
| BLEU | N/A | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | N/A | >= 0.72 | — |
| BERTScore | N/A | >= 0.55 | — |
| COMET-Kiwi | N/A | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 4010.2 days
- **Parametric 95% CI**: [2646.7, 5373.7] days
- **GPU hours**: 192489.3
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
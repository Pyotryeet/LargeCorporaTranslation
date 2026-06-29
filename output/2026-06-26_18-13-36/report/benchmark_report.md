# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T18:14:19Z

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
| Mean tokens/second | 21805.1 |
| Median tokens/second | 21805.5 |
| P5 tokens/second | 21654.0 |
| P95 tokens/second | 21906.0 |
| Std dev | 116.8 |
| Total output tokens | 656,640 |
| Total batches | 5 |

## GPU Utilisation

| Metric | Value |
|---|---|
| Mean GPU utilisation | 47.2 % |
| P99 GPU utilisation | 98.0 % |
| Data starvation (<20%) | 51.6 % |
| Mean GPU temperature | 41.1 °C |

## Quality Scores

| Metric | Score | Target | Status |
|---|---|---|---|
| BLEU | N/A | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | N/A | >= 0.72 | — |
| BERTScore | N/A | >= 0.55 | — |
| COMET-Kiwi | N/A | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 3306.8 days
- **Bootstrap 95% CI**: [3294.7, 3321.2] days (5 batches)
- **Parametric 95% CI**: [3284.8, 3328.8] days
- **GPU hours**: 158726.5
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
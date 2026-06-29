# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T18:03:40Z

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
| Mean tokens/second | 21751.6 |
| Median tokens/second | 21821.4 |
| P5 tokens/second | 21557.6 |
| P95 tokens/second | 21846.4 |
| Std dev | 140.6 |
| Total output tokens | 656,640 |
| Total batches | 5 |

## GPU Utilisation

| Metric | Value |
|---|---|
| Mean GPU utilisation | 47.3 % |
| P99 GPU utilisation | 98.0 % |
| Data starvation (<20%) | 51.6 % |
| Mean GPU temperature | 40.9 °C |

## Quality Scores

| Metric | Score | Target | Status |
|---|---|---|---|
| BLEU | N/A | >= 25 | — |
| chrF++ | N/A | >= 54 | — |
| COMET-22 | N/A | >= 0.72 | — |
| BERTScore | N/A | >= 0.55 | — |
| COMET-Kiwi | N/A | >= 0.72 | — |

## Extrapolation

- **Point estimate**: 3304.4 days
- **Bootstrap 95% CI**: [3301.7, 3334.5] days (5 batches)
- **Parametric 95% CI**: [3277.9, 3330.9] days
- **GPU hours**: 158610.9
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
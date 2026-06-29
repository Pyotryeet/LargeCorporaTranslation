# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T17:28:00Z

## Configuration

- **Backend**: auto
- **Model**: facebook/nllb-200-3.3B
- **Target Duration**: 120 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 14808.8 |
| Median tokens/second | 15108.5 |
| P5 tokens/second | 13535.4 |
| P95 tokens/second | 15192.8 |
| Std dev | 855.4 |
| Total output tokens | 1,838,592 |
| Total batches | 7 |

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

- **Point estimate**: 4772.6 days
- **Bootstrap 95% CI**: [4757.9, 5093.7] days (7 batches)
- **Parametric 95% CI**: [4517.6, 5027.5] days
- **GPU hours**: 229083.7
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
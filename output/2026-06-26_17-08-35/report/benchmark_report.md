# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T17:10:55Z

## Configuration

- **Backend**: auto
- **Model**: facebook/nllb-200-distilled-600M
- **Target Duration**: 120 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 42936.5 |
| Median tokens/second | 42935.9 |
| P5 tokens/second | 42657.0 |
| P95 tokens/second | 43180.2 |
| Std dev | 228.0 |
| Total output tokens | 5,253,120 |
| Total batches | 20 |

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

- **Point estimate**: 1679.4 days
- **Bootstrap 95% CI**: [1675.9, 1683.6] days (20 batches)
- **Parametric 95% CI**: [1675.2, 1683.6] days
- **GPU hours**: 80611.1
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
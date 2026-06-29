# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T17:25:08Z

## Configuration

- **Backend**: auto
- **Model**: facebook/nllb-200-3.3B
- **Target Duration**: 300 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 7527.0 |
| Median tokens/second | 7656.8 |
| P5 tokens/second | 7019.6 |
| P95 tokens/second | 7659.3 |
| Std dev | 343.7 |
| Total output tokens | 919,296 |
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

- **Point estimate**: 9417.3 days
- **Bootstrap 95% CI**: [9415.7, 9922.6] days (7 batches)
- **Parametric 95% CI**: [9019.6, 9815.0] days
- **GPU hours**: 452031.0
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T17:13:30Z

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
| Mean tokens/second | 46845.1 |
| Median tokens/second | 46848.8 |
| P5 tokens/second | 46550.4 |
| P95 tokens/second | 47041.7 |
| Std dev | 192.9 |
| Total output tokens | 5,778,432 |
| Total batches | 11 |

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

- **Point estimate**: 1539.1 days
- **Bootstrap 95% CI**: [1536.1, 1543.2] days (11 batches)
- **Parametric 95% CI**: [1534.9, 1543.4] days
- **GPU hours**: 73878.3
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
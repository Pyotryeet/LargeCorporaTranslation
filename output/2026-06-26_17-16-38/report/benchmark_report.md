# Turkish Corpus Translation Benchmark — Report

**Generated**: 2026-06-26T17:19:20Z

## Configuration

- **Backend**: auto
- **Model**: facebook/nllb-200-distilled-1.3B
- **Target Duration**: 300 s
- **Seed**: 42

## Environment

- **Backend**: N/A
- **Device**: N/A
- **PyTorch**: 2.12.1+cu126

## Throughput Summary

| Metric | Value |
|---|---|
| Mean tokens/second | 12368.7 |
| Median tokens/second | 12386.6 |
| P5 tokens/second | 12304.1 |
| P95 tokens/second | 12388.5 |
| Std dev | 44.8 |
| Total output tokens | 1,575,936 |
| Total batches | 6 |

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

- **Point estimate**: 5821.3 days
- **Bootstrap 95% CI**: [5820.8, 5847.1] days (6 batches)
- **Parametric 95% CI**: [5799.2, 5843.5] days
- **GPU hours**: 279423.8
- **Cost estimate**: $N/A

## Caveats

- Extrapolation assumes constant throughput and 24/7 operation.
- The input sample may not be perfectly representative of the full corpus.
- Thermal throttling may reduce throughput over longer runs.
"""Entrypoint — python -m benchmark.

Full CLI reference (v3.6)
-------------------------
  python -m benchmark --config config.yaml            Full run
  python -m benchmark --config config.yaml --dry-run  10-batch smoke test
  python -m benchmark --config config.yaml --quick    5-minute evaluation
  python -m benchmark --config config.yaml --warmup-only  Warm-up then exit
  python -m benchmark --config config.yaml --benchmark-only  Quality only
  python -m benchmark --config config.yaml --translate-only   Skip quality
  python -m benchmark --config config.yaml --resume <dir>     Resume
  python -m benchmark --config config.yaml --batch-size 128   Force batch size
  python -m benchmark --config config.yaml --no-compile        Disable torch.compile
"""

import argparse
import sys
import warnings
import os as _os

# ── Suppress expected third-party warnings ─────────────────────────────
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*",
                        category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message=".*CUDA Graph is empty.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*",
                        category=UserWarning)
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from benchmark.orchestration.harness import BenchmarkHarness


def main():
    parser = argparse.ArgumentParser(
        description="Turkish Corpus Translation Benchmark Harness v3.6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark --config config.yaml
  python -m benchmark --config config.yaml --dry-run
  python -m benchmark --config config.yaml --quick
  python -m benchmark --config config.yaml --warmup-only
  python -m benchmark --config config.yaml --benchmark-only
  python -m benchmark --config config.yaml --translate-only
  python -m benchmark --config config.yaml --resume output/2026-06-21_14-32-00/
  python -m benchmark --config config.yaml --batch-size 128
  python -m benchmark --config config.yaml --no-compile --safe-mode
        """,
    )

    # ── Config ──
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")

    # ── Run modes ──
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true",
                            help="10-batch smoke test, then exit")
    mode_group.add_argument("--quick", action="store_true",
                            help="5-minute evaluation run")
    mode_group.add_argument("--warmup-only", action="store_true",
                            help="Load model + warm-up only, then exit")
    mode_group.add_argument("--benchmark-only", action="store_true",
                            help="Skip translation, run quality benchmark only")
    mode_group.add_argument("--translate-only", action="store_true",
                            help="Skip quality benchmark after translation")

    # ── Resume ──
    parser.add_argument("--resume", metavar="DIR",
                        help="Resume from a checkpoint directory")

    # ── Overrides ──
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Force a specific batch size (0 = auto-tune, default)")
    parser.add_argument("--duration", type=int,
                        help="Override target_duration_seconds from config")
    parser.add_argument("--seed", type=int,
                        help="Override random seed from config")

    # ── v3.6 flags ──
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (useful for debugging)")
    parser.add_argument("--safe-mode", action="store_true",
                        help="Disable CUDA graph, paged attn, TE FP8, fused kernels "
                             "— safest path for correctness")
    parser.add_argument("--observability", action="store_true",
                        help="Enable Prometheus metrics server on localhost:9090")
    parser.add_argument("--speculative", action="store_true",
                        help="Enable speculative decoding (self-speculative by default, "
                             "uses early-layer draft — zero extra VRAM)")
    parser.add_argument("--spec-mode", choices=["self", "draft_model"], default="self",
                        help="Speculative mode: 'self' (early-layer) or 'draft_model'")
    parser.add_argument("--spec-tokens", type=int, default=3,
                        help="Number of speculative tokens K (default: 3)")
    parser.add_argument("--spec-draft-layers", type=int, default=0,
                        help="Early layers for self-spec draft (0=auto)")
    parser.add_argument("--paged-attention", action="store_true",
                        help="Enable PagedAttention KV-cache (40-70%% less memory, CUDA only)")
    parser.add_argument("--continuous-batching", action="store_true",
                        help="Enable continuous batching (higher throughput, CUDA only)")
    parser.add_argument("--nllb", action="store_true",
                        help="Use encoder-decoder backend for NLLB translation models")
    parser.add_argument("--nllb-src-lang", type=str, default="eng_Latn",
                        help="Source language code for NLLB (default: eng_Latn)")
    parser.add_argument("--nllb-tgt-lang", type=str, default="tur_Latn",
                        help="Target language code for NLLB (default: tur_Latn)")
    parser.add_argument("--mps-safe", action="store_true",
                        help="On Apple Silicon: skip batch tuning (bs=1), skip shuffle, "
                             "minimise IOAccelerator pressure (default on, set "
                             "TR_MPS_MEMORY_SAFE=0 to disable)")

    parser.add_argument("--quantization", choices=["bf16", "fp16", "int8", "int4"], default=None,
                        help="Model quantization level: bf16 (unquantized), int8 (8-bit), "
                             "int4 (4-bit NF4)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model preset name or HF model ID (e.g. '4B', 'ministral-3b', "
                             "'google/gemma-4-E2B-it-qat-mobile-ct')")

    args = parser.parse_args()

    # ── Apply MPS memory-safe mode before harness creation ──
    if args.mps_safe:
        import os as _os
        _os.environ["TR_MPS_MEMORY_SAFE"] = "1"

    # ── Inject CLI overrides into config (speculative, paged attention, NLLB, etc.) ──
    _needs_inject = (
        args.speculative or args.paged_attention or args.continuous_batching
        or args.nllb or args.quantization is not None or args.model is not None
    )
    if _needs_inject:
        import yaml
        with open(args.config, "r") as _f:
            _cfg = yaml.safe_load(_f) or {}
        _model = _cfg.setdefault("model", {})

        if args.speculative:
            _model["use_speculative"] = True
            _model["speculative_mode"] = args.spec_mode if args.spec_mode != "self" else "self"
            _model["speculative_num_tokens"] = args.spec_tokens
            _model["speculative_num_draft_layers"] = args.spec_draft_layers

        if args.paged_attention:
            _model["use_paged_attention"] = True

        if args.continuous_batching:
            _model["use_continuous_batching"] = True

        if args.nllb:
            _model["backend_type"] = "encoder_decoder"
            _model["nllb_source_lang"] = args.nllb_src_lang
            _model["nllb_target_lang"] = args.nllb_tgt_lang

        if args.quantization is not None:
            _model["quantization"] = args.quantization

        if args.model is not None:
            _model["model_path"] = args.model

        # Write a temporary config so the harness picks up the injected values.
        import tempfile
        import atexit
        _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(_cfg, _tmp)
        _tmp.close()
        atexit.register(lambda p=_tmp.name: __import__('os').unlink(p) if __import__('os').path.exists(p) else None)
        args.config = _tmp.name

    # Determine the effective run mode
    if args.resume:
        run_mode = "resume"
    elif args.warmup_only:
        run_mode = "warmup-only"
    elif args.translate_only:
        run_mode = "translate-only"
    elif args.dry_run:
        run_mode = "dry-run"
    elif args.quick:
        run_mode = "quick"
    elif args.benchmark_only:
        run_mode = "benchmark-only"
    else:
        run_mode = "full"

    harness = BenchmarkHarness(
        config_path=args.config,
        run_mode=run_mode,
        batch_size_override=args.batch_size if args.batch_size > 0 else None,
        duration_override=args.duration,
        seed_override=args.seed,
        resume_dir=args.resume,
        no_torch_compile=args.no_compile,
        safe_mode=args.safe_mode,
        observability_enabled=args.observability,
    )

    report = harness.run()

    # ── Summary ──
    print("\n=== Benchmark Complete ===")
    ext = report.get("extrapolation", {})
    if ext.get("days_point_estimate"):
        print(
            "Estimated: {:.1f} days  (95% CI: {:.1f}–{:.1f})".format(
                ext["days_point_estimate"],
                ext["days_95ci_lower"],
                ext["days_95ci_upper"],
            )
        )
    quality = report.get("quality", {})
    if quality:
        bertscore = quality.get("bertscore", {}).get("system_score", "N/A")
        comet = quality.get("comet", {}).get("system_score", "N/A")
        comet_kiwi = quality.get("comet_kiwi", {}).get("system_score", "N/A")
        print(f"BERTScore: {bertscore} | COMET-22: {comet} | COMET-Kiwi: {comet_kiwi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

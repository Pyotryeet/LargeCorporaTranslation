#!/usr/bin/env python3
"""Tune metric weights from human evaluation feedback.

Loads human ratings + automated metrics, performs linear regression
to learn optimal weights for the composite accuracy score, ranks models,
and selects the best candidate for production.

Input:
  - data/output/model_selection/human_eval_data.json  (sources + label maps)
  - data/output/model_selection/metrics.json          (automated scores)
  - benchmark_sonuclari.json                          (human ratings, from index.html)

Output:
  - data/output/model_selection/weights.json          (learned weights + R²)
  - data/output/model_selection/final_ranking.json    (model ranking)
"""
import json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "model_selection"
HUMAN_RATINGS_FILE = Path("benchmark_sonuclari.json")  # from index.html download
METRICS_FILE = OUTPUT_DIR / "metrics.json"
HUMAN_EVAL_DATA = OUTPUT_DIR / "human_eval_data.json"

METRIC_KEYS = [
    "bleurt", "comet", "comet_kiwi", "metricx_24",
    "chrf_seg",
    # System-level metrics (applied to all segments of a model)
    "chrf", "chrf_pp", "bleu", "sacrbleu", "spbleu", "morph_bleu",
]

SEGMENT_METRICS = ["bleurt", "comet", "comet_kiwi", "metricx_24", "chrf_seg"]
SYSTEM_METRICS = ["chrf", "chrf_pp", "bleu", "sacrbleu", "spbleu", "morph_bleu"]


def main():
    # ── Load data ───────────────────────────────────────────────────────
    if not HUMAN_RATINGS_FILE.exists():
        print(f"ERROR: {HUMAN_RATINGS_FILE} not found.")
        print("Download this from the index.html web interface after completing evaluations.")
        sys.exit(1)

    with open(HUMAN_RATINGS_FILE, "r", encoding="utf-8") as f:
        human_data = json.load(f)

    with open(HUMAN_EVAL_DATA, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics_list = json.load(f)

    # Build lookups
    # human_data.ratings = [{source_id, source_text, ratings: {label: score}}, ...]
    # eval_data.sources = [{id, label_map: {label: model_id}}, ...]
    # metrics = [{source_id, model_id, metrics: {key: val}}, ...]

    # Build: model_id → system-level metrics
    model_ids = eval_data["model_ids"]
    system_metrics = {}
    for item in metrics_list:
        mid = item["model_id"]
        if mid not in system_metrics:
            system_metrics[mid] = {}
        for key in SYSTEM_METRICS:
            val = item["metrics"].get(key)
            if val is not None:
                system_metrics[mid][key] = val

    # Build: label → model_id per source
    label_to_model = {}
    for src in eval_data["sources"]:
        label_to_model[src["id"]] = src.get("label_map", {})

    # Build: (source_id, model_id) → segment metrics
    seg_metrics = {}
    for item in metrics_list:
        sid = item["source_id"]
        mid = item["model_id"]
        seg_metrics[(sid, mid)] = {
            k: v for k, v in item["metrics"].items()
            if v is not None and k in SEGMENT_METRICS
        }

    # ── Build training data ─────────────────────────────────────────────
    # For each human rating, collect the corresponding segment metrics
    X_rows = []  # feature vectors
    y_scores = []  # human ratings

    for rating_entry in human_data.get("ratings", []):
        sid = rating_entry["source_id"]
        for label, score in rating_entry.get("ratings", {}).items():
            mid = label_to_model.get(sid, {}).get(label)
            if mid is None:
                print(f"  WARNING: source {sid} label {label} → model not found")
                continue

            # Collect segment metrics
            seg = seg_metrics.get((sid, mid), {})
            # Collect system metrics for this model
            sys = system_metrics.get(mid, {})

            feature = []
            for key in SEGMENT_METRICS:
                feature.append(seg.get(key, np.nan))
            for key in SYSTEM_METRICS:
                feature.append(sys.get(key, np.nan))

            X_rows.append(feature)
            y_scores.append(score)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_scores, dtype=float)

    print(f"Training data: {len(y)} human ratings from {len(human_data.get('ratings', []))} sources")
    print(f"Features per sample: {X.shape[1]} ({len(SEGMENT_METRICS)} segment + {len(SYSTEM_METRICS)} system)")

    # ── Impute missing values with column means ─────────────────────────
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]

    # ── Normalize features to [0, 1] ────────────────────────────────────
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # avoid division by zero
    X_norm = (X - X_min) / X_range

    # ── Linear regression ───────────────────────────────────────────────
    # y = X @ w + b
    X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])  # add bias
    w, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)

    weights = w[:-1]
    bias = w[-1]
    y_pred = X_aug @ w
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    all_feature_names = SEGMENT_METRICS + SYSTEM_METRICS
    print(f"\nLinear regression results:")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Bias = {bias:.4f}")
    print(f"\n  Feature weights (normalized, higher = more predictive of human score):")
    for name, w_val in sorted(zip(all_feature_names, weights), key=lambda x: -abs(x[1])):
        bar = "█" * max(1, int(abs(w_val) * 40))
        print(f"    {name:>16s}: {w_val:+.4f}  {bar}")

    # ── Compute composite scores per model ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"MODEL RANKING (composite weighted score)")
    print(f"{'='*60}")

    model_scores = {}
    for mid in model_ids:
        scores_for_model = []
        for sid in range(30):  # 30 sources
            seg = seg_metrics.get((sid, mid), {})
            sys = system_metrics.get(mid, {})
            feature = []
            for key in SEGMENT_METRICS:
                val = seg.get(key, col_means[len(feature)] if len(feature) < len(col_means) else 0)
                feature.append(val if val is not None and not np.isnan(val) else 0)
            for key in SYSTEM_METRICS:
                val = sys.get(key, col_means[len(feature)] if len(feature) < len(col_means) else 0)
                feature.append(val if val is not None and not np.isnan(val) else 0)

            feature_arr = np.array(feature, dtype=float)
            # Impute + normalize using stored min/max
            for j in range(len(feature_arr)):
                if np.isnan(feature_arr[j]):
                    feature_arr[j] = col_means[j]
            feature_norm = (feature_arr - X_min) / X_range
            composite = np.dot(feature_norm, weights) + bias
            scores_for_model.append(composite)

        model_scores[mid] = {
            "mean": float(np.mean(scores_for_model)),
            "std": float(np.std(scores_for_model)),
            "min": float(np.min(scores_for_model)),
            "max": float(np.max(scores_for_model)),
        }

    # Rank models
    ranked = sorted(model_scores.items(), key=lambda x: x[1]["mean"], reverse=True)
    model_names = {
        "nllb_600m": "NLLB-200 600M",
        "nllb_1.3b": "NLLB-200 1.3B",
        "nllb_3.3b": "NLLB-200 3.3B",
        "madlad_3b": "MADLAD-400 3B",
        "translategemma_4b": "TranslateGemma 4B",
        "smollm2_1.7b": "SmolLM2 1.7B",
    }
    print(f"\n{'Rank':<5s} {'Model':<25s} {'Score':>8s}  {'±Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"{'-'*5} {'-'*25} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for rank, (mid, scores) in enumerate(ranked, 1):
        name = model_names.get(mid, mid)
        marker = " ← BEST" if rank == 1 else ""
        print(f"{rank:<5d} {name:<25s} {scores['mean']:>8.4f}  ±{scores['std']:>7.4f}  {scores['min']:>8.4f}  {scores['max']:>8.4f}{marker}")

    # ── Save results ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weights_output = {
        "r_squared": round(float(r_squared), 4),
        "bias": round(float(bias), 4),
        "n_samples": len(y),
        "weights": {name: round(float(w), 4) for name, w in zip(all_feature_names, weights)},
        "normalization": {
            "min": [float(x) for x in X_min],
            "max": [float(x) for x in X_max],
        },
    }
    with open(OUTPUT_DIR / "weights.json", "w") as f:
        json.dump(weights_output, f, indent=2)
    print(f"\nWeights saved to {OUTPUT_DIR / 'weights.json'}")

    ranking_output = {
        "ranking": [
            {
                "rank": i + 1,
                "model_id": mid,
                "model_name": model_names.get(mid, mid),
                "mean_score": round(scores["mean"], 4),
                "std_score": round(scores["std"], 4),
                "is_best": i == 0,
            }
            for i, (mid, scores) in enumerate(ranked)
        ],
        "best_model": {
            "model_id": ranked[0][0],
            "model_name": model_names.get(ranked[0][0], ranked[0][0]),
            "config_path": ranked[0][0],
        },
    }
    with open(OUTPUT_DIR / "final_ranking.json", "w") as f:
        json.dump(ranking_output, f, indent=2)
    print(f"Ranking saved to {OUTPUT_DIR / 'final_ranking.json'}")

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {ranking_output['best_model']['model_name']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

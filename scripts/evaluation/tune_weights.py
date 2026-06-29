#!/usr/bin/env python3
"""Tune metric weights from human evaluation feedback.

Loads human ratings + automated metrics, performs linear regression
at the model level to learn optimal weights for the composite accuracy score,
ranks models, and selects the best candidate for production.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "model_selection"
HUMAN_RATINGS_FILE = Path("benchmark_sonuclari.json")  # from index.html download
METRICS_FILE = OUTPUT_DIR / "metrics.json"
HUMAN_EVAL_DATA = OUTPUT_DIR / "human_eval_data.json"

METRIC_KEYS = [
    "comet_kiwi", "comet", "metricx_24", "bertscore", "chrf", "spbleu", "morph_bleu"
]


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

    # 1. Build label to model mapping per source
    label_to_model = {}
    for src in eval_data.get("sources", []):
        sid = src.get("id")
        label_to_model[sid] = src.get("label_map", {})

    # 2. Collect human ratings per model
    model_ratings = {}
    total_ratings_count = 0
    for rating_entry in human_data.get("ratings", []):
        sid = rating_entry.get("source_id")
        for label, score in rating_entry.get("ratings", {}).items():
            mid = label_to_model.get(sid, {}).get(label)
            if mid is not None:
                model_ratings.setdefault(mid, []).append(float(score))
                total_ratings_count += 1
            else:
                print(f"  WARNING: source {sid} label {label} → model not found")

    if not model_ratings:
        print("ERROR: No matching human ratings could be mapped to model IDs.")
        sys.exit(1)

    # Compute average human rating per model
    model_avg_human = {}
    for mid, ratings in model_ratings.items():
        model_avg_human[mid] = np.mean(ratings)
        print(f"Model {mid}: average human rating = {model_avg_human[mid]:.2f} (from {len(ratings)} scores)")

    # 3. Extract metrics per model
    model_metrics = {}
    for item in metrics_list:
        mid = item["model_id"]
        # Ensure we have ratings for this model
        if mid in model_avg_human:
            model_metrics[mid] = item["metrics"]

    # ── Build Regression Dataset ─────────────────────────────────────────
    # We fit a regression at the model level: y (avg human rating) = X (metrics) @ w + b
    model_ids = sorted(list(model_avg_human.keys()))
    X_rows = []
    y_scores = []

    for mid in model_ids:
        metrics = model_metrics.get(mid, {})
        row = [float(metrics.get(key, 0.0)) for key in METRIC_KEYS]
        X_rows.append(row)
        y_scores.append(model_avg_human[mid])

    X = np.array(X_rows, dtype=float)
    y = np.array(y_scores, dtype=float)

    # Normalize metrics to [0, 1] range to make learned weights comparable
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0  # prevent division by zero
    X_norm = (X - X_min) / X_range

    # Linear Regression: y = X_norm @ w + b
    # Add bias term column
    X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])
    w, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)

    weights = w[:-1]
    bias = w[-1]
    y_pred = X_aug @ w
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f"\nLinear regression results (Model-level regression):")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Bias = {bias:.4f}")
    print(f"\n  Feature weights (normalized, higher = more predictive of human score):")
    for name, w_val in sorted(zip(METRIC_KEYS, weights), key=lambda x: -abs(x[1])):
        bar = "█" * max(1, int(abs(w_val) * 40))
        print(f"    {name:>16s}: {w_val:+.4f}  {bar}")

    # ── Compute composite scores per model ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"MODEL RANKING (composite weighted score)")
    print(f"{'='*60}")

    model_scores = {}
    for mid in model_ids:
        metrics = model_metrics.get(mid, {})
        row = np.array([float(metrics.get(key, 0.0)) for key in METRIC_KEYS], dtype=float)
        row_norm = (row - X_min) / X_range
        composite = float(np.dot(row_norm, weights) + bias)
        
        # Calculate standard deviation/range based on individual ratings if available
        ratings = model_ratings.get(mid, [composite])
        model_scores[mid] = {
            "mean": composite,
            "std": float(np.std(ratings)),
            "min": float(np.min(ratings)),
            "max": float(np.max(ratings)),
        }

    # Rank models
    ranked = sorted(model_scores.items(), key=lambda x: x[1]["mean"], reverse=True)
    model_names = {
        "nllb_600m": "NLLB-200 600M",
        "nllb_1.3b": "NLLB-200 1.3B",
        "nllb_3.3b": "NLLB-200 3.3B",
        "nllb_moe_54b": "NLLB-200 MoE 54B",
        "madlad_3b": "MADLAD-400 3B",
        "madlad_10b": "MADLAD-400 10B",
        "translategemma_4b": "TranslateGemma 4B",
        "translategemma_12b": "TranslateGemma 12B",
        "translategemma_27b": "TranslateGemma 27B",
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
        "n_samples": total_ratings_count,
        "weights": {name: round(float(w), 4) for name, w in zip(METRIC_KEYS, weights)},
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

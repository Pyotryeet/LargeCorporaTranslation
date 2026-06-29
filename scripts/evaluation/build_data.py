#!/usr/bin/env python3
"""Build human evaluation data JSON for the web interface.

Loads translations + metrics, anonymizes model names (A-F),
randomizes display order per source for blind testing.
"""
import json, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_TRANSLATIONS = PROJECT_ROOT / "data" / "output" / "model_selection" / "translations.json"
INPUT_METRICS = PROJECT_ROOT / "data" / "output" / "model_selection" / "metrics.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "model_selection" / "human_eval_data.json"

MODEL_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


def main():
    with open(INPUT_TRANSLATIONS, "r", encoding="utf-8") as f:
        translations = json.load(f)

    # Build model mapping: model_id → label
    model_ids = list(translations[0]["models"].keys())
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

    # Load metrics if available
    metrics_data = None
    if INPUT_METRICS.exists():
        with open(INPUT_METRICS, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        # Build lookup: (source_id, model_id) → metrics
        metrics_lookup = {}
        for item in metrics_data:
            sid = item.get("source_id")
            mid = item.get("model_id")
            if sid is not None and mid is not None:
                key = (sid, mid)
                metrics_lookup[key] = item.get("metrics", {})

    rng = random.Random(42)
    sources = []

    for entry in translations:
        sid = entry["source_id"]
        src_text = entry["source_text"]

        # Randomize model order for blind testing
        shuffled = list(model_ids)
        rng.shuffle(shuffled)
        # Assign labels consistently within this source
        source_labels = {mid: MODEL_LABELS[i] for i, mid in enumerate(shuffled)}

        trans_list = []
        for mid in shuffled:
            m = entry["models"][mid]
            label = source_labels[mid]
            item = {
                "model_label": label,
                "text": m["text"],
                "output_tokens": m.get("output_tokens", 0),
            }
            # Embed automated metrics (hidden from evaluator)
            if metrics_lookup:
                mets = metrics_lookup.get((sid, mid), {})
                if mets:
                    item["_metrics"] = {
                        k: v for k, v in mets.items()
                        if v is not None and k != "chrf_seg"
                    }
            trans_list.append(item)

        sources.append({
            "id": sid,
            "text": src_text,
            "char_len": entry.get("source_char_len", len(src_text)),
            "translations": trans_list,
            "label_map": {label: mid for mid, label in source_labels.items()},
        })

    output = {
        "sources": sources,
        "model_mapping": {label: model_names.get(mid, mid)
                          for mid, label in sorted(
                              [(mid, list(source_labels.values())[list(source_labels.keys()).index(mid)])
                               for mid in model_ids],
                              key=lambda x: x[1],
                          )},
        "model_ids": model_ids,
        "total_sources": len(sources),
        "models_per_source": len(model_ids),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sources)} sources × {len(model_ids)} models to {OUTPUT_FILE}")
    print(f"Model labels: {output['model_mapping']}")


if __name__ == "__main__":
    main()

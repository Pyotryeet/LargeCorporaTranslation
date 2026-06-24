#!/usr/bin/env python3
"""Sentence selection via embedding + k-means clustering.

Loads 5,000 documents from the FineWeb sample, extracts sentences,
embeds them with bert-base-multilingual-cased, clusters into 30 groups,
and picks the centroid sentence from each cluster.

Output: data/output/model_selection/source_sentences.json
"""
import gzip, json, os, sys, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "input" / "fineweb_en_sample.jsonl.gz"
OUTPUT_DIR = DATA_DIR / "output" / "model_selection"
OUTPUT_FILE = OUTPUT_DIR / "source_sentences.json"
N_DOCS = 5_000
N_CLUSTERS = 30
MIN_SENTENCE_LEN = 40   # skip very short sentences
MAX_SENTENCE_LEN = 300  # skip very long sentences (they're often garbage)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {N_DOCS} documents from {INPUT_FILE}...")
    docs = []
    with gzip.open(INPUT_FILE, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= N_DOCS:
                break
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if text:
                    docs.append((i, text))
            except Exception:
                pass
    print(f"  Loaded {len(docs)} documents")

    # ── Sentence tokenization ───────────────────────────────────────────
    print("Tokenizing sentences...")
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    sentences = []
    for doc_id, text in docs:
        raw = nltk.sent_tokenize(text)
        for sent in raw:
            s = sent.strip()
            if MIN_SENTENCE_LEN <= len(s) <= MAX_SENTENCE_LEN:
                sentences.append({
                    "text": s,
                    "doc_id": doc_id,
                    "char_len": len(s),
                })

    print(f"  Extracted {len(sentences)} sentences "
          f"({MIN_SENTENCE_LEN}-{MAX_SENTENCE_LEN} chars)")

    # ── Sentence embeddings ─────────────────────────────────────────────
    print("Computing sentence embeddings...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("bert-base-multilingual-cased")
    texts = [s["text"] for s in sentences]
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    elapsed = time.time() - start
    print(f"  Embedded {len(texts)} sentences in {elapsed:.1f}s "
          f"({len(texts)/elapsed:.0f}/s)")

    # ── K-means clustering ──────────────────────────────────────────────
    print(f"Clustering into {N_CLUSTERS} groups...")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # ── Pick centroids ──────────────────────────────────────────────────
    print("Selecting centroid sentences...")
    selected = []
    for cluster_id in range(N_CLUSTERS):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_centroid = kmeans.cluster_centers_[cluster_id]
        # Find sentence closest to centroid
        cluster_embeddings = embeddings[cluster_mask]
        distances = np.linalg.norm(cluster_embeddings - cluster_centroid, axis=1)
        best_local_idx = int(np.argmin(distances))
        best_global_idx = int(cluster_indices[best_local_idx])
        sent = sentences[best_global_idx]
        selected.append({
            "id": cluster_id,
            "text": sent["text"],
            "doc_id": sent["doc_id"],
            "cluster": cluster_id,
            "char_len": sent["char_len"],
            "cluster_size": int(cluster_mask.sum()),
        })

    # Sort by char_len for diversity display
    selected.sort(key=lambda s: s["char_len"])

    print(f"\n  Selected {len(selected)} sentences:")
    print(f"  {'Cluster':>8s} {'Len':>5s} {'Size':>6s}  Text")
    print(f"  {'-'*8} {'-'*5} {'-'*6}  {'-'*40}")
    for s in selected:
        print(f"  {s['cluster']:>8d} {s['char_len']:>5d} {s['cluster_size']:>6d}  {s['text'][:70]}...")

    lengths = [s["char_len"] for s in selected]
    print(f"\n  Length distribution: min={min(lengths)} max={max(lengths)} "
          f"mean={np.mean(lengths):.0f} std={np.std(lengths):.0f}")

    # ── Save ────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(selected)} sentences to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

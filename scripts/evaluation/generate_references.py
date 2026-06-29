#!/usr/bin/env python3
"""Generate reference translations using Gemini 3.1 Pro (thinking=high).

Uses the official google-genai SDK (pip install -U google-genai).
Requires GOOGLE_API_KEY environment variable set to a valid API key.

Output: data/output/model_selection/references.jsonl
"""
import json, os
from pathlib import Path

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not API_KEY:
    print("ERROR: Set GOOGLE_API_KEY environment variable")
    print("Get a key at https://aistudio.google.com")
    exit(1)

ROOT = Path(__file__).resolve().parent.parent.parent
INFILE = ROOT / "data" / "output" / "model_selection" / "source_sentences.json"
OUTFILE = ROOT / "data" / "output" / "model_selection" / "references.jsonl"
OUTFILE.parent.mkdir(parents=True, exist_ok=True)

with open(INFILE) as f:
    sentences = json.load(f)
print(f"Sentences: {len(sentences)}")

from google import genai
from google.genai import types

client = genai.Client(api_key=API_KEY)

results = []
for i, s in enumerate(sentences):
    sys_instruction = (
        "You are an expert, professional English-to-Turkish translator with deep knowledge of Turkish grammar, "
        "idioms, agglutinative morphology, and natural syntax. "
        "Your task is to translate the source text into natural, fluent, and highly accurate Turkish. "
        "Fidelity is paramount: match the exact meaning, detail, tone, and formatting of the source. "
        "Use natural Turkish structures and vocabulary suited to the context. "
        "Strictly output ONLY the translation itself. Do not include any notes, explanations, thinking steps, "
        "or introductory/concluding text in the final output."
    )
    model_used = "gemini-3.1-pro-preview"
    try:
        response = client.models.generate_content(
            model=model_used,
            contents=s["text"],
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=0.0,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(
                    thinking_level="high"
                )
            ),
        )
        text = response.text.strip()
    except Exception as e:
        print(f"  [{s['id']:>2d}] ERROR: {e}")
        text = ""

    results.append({
        "source_text": s["text"],
        "reference_translation": text,
        "source": model_used,
    })
    print(f"  [{s['id']:>2d}] {text[:100]}...")

with open(OUTFILE, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n{len(results)} references → {OUTFILE}")

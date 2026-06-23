"""Golden reference set loader."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
MIN_REFERENCE_PAIRS = 10


class ReferenceLoader:
    def __init__(self, reference_path: str | Path):
        self.reference_path = Path(reference_path)

    def load(self) -> tuple[list[str], list[str]]:
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {self.reference_path}")
        sources = []
        references = []
        with open(self.reference_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # NOTE: The or-chaining below only works correctly for string values.
                # If any field could be an int 0 or bool False, a falsy check would
                # discard the value incorrectly — use explicit None checks in that case.
                src = obj.get("source_text") or obj.get("src") or obj.get("en", "")
                ref = obj.get("reference_translation") or obj.get("ref") or obj.get("tr", "")
                if src and ref:
                    sources.append(src)
                    references.append(ref)
        logger.info("Loaded %d reference pairs from %s", len(sources), self.reference_path)
        if len(sources) < MIN_REFERENCE_PAIRS:
            logger.warning("Only %d reference pairs — benchmark may be unreliable", len(sources))
        return sources, references

    @staticmethod
    def validate_pair(source: str, reference: str) -> bool:
        return bool(source and reference and len(source) > 2 and len(reference) > 2)

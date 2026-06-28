"""Golden reference set loader."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
MIN_REFERENCE_PAIRS = 10


class ReferenceLoader:
    """Load and validate golden reference translation pairs from a JSON-lines file.

    Reads a JSON-lines file where each record contains a source text and a
    reference translation. Attempts to auto-detect field names ("source_text" /
    "reference_translation", "src" / "ref", "en" / "tr"). Rejects pairs that
    fail minimum-quality validation. Raises ValueError if zero valid pairs are
    found; emits a warning if fewer than MIN_REFERENCE_PAIRS (10) are loaded.
    """

    def __init__(self, reference_path: str | Path):
        """Initialize the loader with a path to a JSON-lines reference file.

        Args:
            reference_path: Path (string or pathlib.Path) to the JSON-lines file
                containing golden reference pairs.
        """
        self.reference_path = Path(reference_path)

    def load(self) -> tuple[list[str], list[str]]:
        """Load and validate reference pairs from the JSON-lines file.

        Iterates over each line in the file, parses JSON, auto-detects source and
        reference fields, validates each pair, and collects the valid ones.

        Returns:
            A tuple of (sources, references) where both are list[str] of equal
            length. Each element at index i in sources corresponds to the
            reference at index i in references.

        Raises:
            FileNotFoundError: If the reference file does not exist.
            ValueError: If no valid (source, reference) pairs are found in the
                file.

        Side effects:
            Logs the number of loaded pairs at INFO level.
            Logs a WARNING if fewer than MIN_REFERENCE_PAIRS (10) are loaded.
        """
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
                # Use explicit None checks instead of or-chaining so that
                # valid empty-string values are not silently discarded.
                src = obj.get("source_text")
                if src is None:
                    src = obj.get("src")
                if src is None:
                    src = obj.get("en")
                if src is None:
                    src = ""

                ref = obj.get("reference_translation")
                if ref is None:
                    ref = obj.get("ref")
                if ref is None:
                    ref = obj.get("tr")
                if ref is None:
                    ref = ""
                if ReferenceLoader.validate_pair(src, ref):
                    sources.append(src)
                    references.append(ref)
        logger.info("Loaded %d reference pairs from %s", len(sources), self.reference_path)
        if len(sources) == 0:
            raise ValueError(
                f"No valid (source, reference) pairs found in {self.reference_path}. "
                f"Check that the file contains JSON records with 'source_text'/"
                f"'reference_translation' fields and non-empty values."
            )
        if len(sources) < MIN_REFERENCE_PAIRS:
            logger.warning("Only %d reference pairs — benchmark may be unreliable", len(sources))
        return sources, references

    @staticmethod
    def validate_pair(source: str, reference: str) -> bool:
        """Return True if the (source, reference) pair passes minimum-quality checks.

        Rejects None, empty strings, non-string types, and strings below the
        minimum length threshold (3 characters).  Whitespace-only strings are
        also rejected because they carry no semantic content.
        """
        if not isinstance(source, str) or not isinstance(reference, str):
            return False
        src_stripped = source.strip()
        ref_stripped = reference.strip()
        return bool(src_stripped and ref_stripped and len(src_stripped) > 2 and len(ref_stripped) > 2)

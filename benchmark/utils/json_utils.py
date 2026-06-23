"""JSON utilities — shared encoder that handles float('inf'), -inf, and NaN.

Standard ``json.dumps`` and ``json.dump`` raise ``ValueError`` when they
encounter non-finite floats.  The encoder provided here maps::

    float('inf')  → 1e308   (unambiguously huge sentinel, still a valid float)
    float('-inf') → -1e308
    float('nan')  → null    (JSON null)

Usage::

    from benchmark.utils.json_utils import sanitized_dumps, sanitized_dump

    sanitized_dumps(any_dict)          # returns str
    sanitized_dump(any_dict, fp)       # writes to file-like object
"""

import json
import math


_SENTINEL_INF = 1e308


class _SanitizingEncoder(json.JSONEncoder):
    """JSONEncoder that sanitises non-finite floats before serialisation."""

    def default(self, o):
        # Let the parent raise TypeError for types we truly cannot serialise.
        return super().default(o)

    def encode(self, o):
        # Recurse through the entire object graph before encoding so that
        # every non-finite float is replaced.
        return super().encode(self._sanitize(o))

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot=_one_shot)

    @classmethod
    def _sanitize(cls, obj):
        """Recursively walk *obj* and replace non-finite floats."""
        if isinstance(obj, float):
            if math.isinf(obj):
                return _SENTINEL_INF if obj > 0 else -_SENTINEL_INF
            if math.isnan(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: cls._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._sanitize(v) for v in obj]
        return obj


def sanitized_dumps(obj, **kwargs) -> str:
    """``json.dumps`` equivalent that handles non-finite floats.

    All keyword arguments are forwarded to ``json.dumps``.
    """
    return json.dumps(obj, cls=_SanitizingEncoder, **kwargs)


def sanitized_dump(obj, fp, **kwargs) -> None:
    """``json.dump`` equivalent that handles non-finite floats.

    All keyword arguments are forwarded to ``json.dump``.
    """
    return json.dump(obj, fp, cls=_SanitizingEncoder, **kwargs)

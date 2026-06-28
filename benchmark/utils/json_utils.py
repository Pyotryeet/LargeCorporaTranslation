"""JSON utilities — shared encoder that handles float('inf'), -inf, and NaN.

Standard ``json.dumps`` and ``json.dump`` raise ``ValueError`` when they
encounter non-finite floats.  The encoder provided here maps::

    float('inf')  → 1e308   (unambiguously huge sentinel, still a valid float)
    float('-inf') → -1e308
    float('nan')  → null    (JSON null)

.. note::

    JSON has no tuple type.  Python ``tuple`` values are preserved through
    the sanitisation walk (non-finite floats within tuples are replaced),
    but the JSON encoder serialises them as JSON arrays.  On round-trip
    through ``json.loads`` they will be ``list`` objects.  Callers that
    need to distinguish tuples and lists should add a post-deserialisation
    conversion step.

.. warning::

    This is a **silent** conversion — no warning is raised when tuples
    become lists after deserialisation.  Any downstream code that relies
    on tuple identity (e.g. ``isinstance(value, tuple)``) will silently
    break after round-tripping through JSON.

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

    def encode(self, o):
        # Recurse through the entire object graph before encoding so that
        # every non-finite float is replaced.
        return super().encode(self._sanitize(o))

    def iterencode(self, o, _one_shot=False):
        # NOTE: iterencode pre-materializes the entire object graph via
        # _sanitize before yielding chunks. This is a deliberate trade-off:
        # a deep copy is created to sanitize non-finite floats, which
        # increases peak memory but is necessary for correctness — many
        # consumers depend on these values being replaced.
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
        if isinstance(obj, tuple):
            return tuple(cls._sanitize(v) for v in obj)
        if isinstance(obj, list):
            return [cls._sanitize(v) for v in obj]
        return obj


def sanitized_dumps(obj, **kwargs) -> str:
    """Serialize *obj* to a JSON string, replacing non-finite floats.

    Parameters
    ----------
    obj : Any
        A JSON-serialisable Python object. May contain ``float('inf')``,
        ``float('-inf')``, or ``float('nan')`` anywhere in the object graph.
    **kwargs
        Additional keyword arguments forwarded directly to ``json.dumps``
        (e.g. ``indent``, ``ensure_ascii``, ``sort_keys``).

    Returns
    -------
    str
        A JSON string where non-finite floats have been replaced:
        ``inf`` → ``1e308``, ``-inf`` → ``-1e308``, ``nan`` → ``null``.

    Notes
    -----
    - Tuples are serialised as JSON arrays. A round-trip through
      ``json.loads`` will produce ``list`` objects, not ``tuple``.
    - The conversion is silent; no warning is raised for the
      tuple-to-list round-trip behaviour or for the non-finite float
      replacements.
    """
    return json.dumps(obj, cls=_SanitizingEncoder, **kwargs)


def sanitized_dump(obj, fp, **kwargs) -> None:
    """Write *obj* as JSON to a file-like object, replacing non-finite floats.

    Parameters
    ----------
    obj : Any
        A JSON-serialisable Python object. May contain ``float('inf')``,
        ``float('-inf')``, or ``float('nan')`` anywhere in the object graph.
    fp : file-like object
        A ``.write()``-supporting file-like object (e.g. ``io.StringIO``,
        a file handle opened in text mode) to which the JSON output will
        be written.
    **kwargs
        Additional keyword arguments forwarded directly to ``json.dump``
        (e.g. ``indent``, ``ensure_ascii``, ``sort_keys``).

    Returns
    -------
    None

    Notes
    -----
    - Tuples are serialised as JSON arrays. A round-trip through
      ``json.load`` will produce ``list`` objects, not ``tuple``.
    - The conversion is silent; no warning is raised for the
      tuple-to-list round-trip behaviour or for the non-finite float
      replacements.
    """
    return json.dump(obj, fp, cls=_SanitizingEncoder, **kwargs)

"""Shared utility helpers used across benchmark modules."""

import os


def local_kwargs(path: str) -> dict:
    """Return ``{"local_files_only": True}`` if *path* is a local file/dir.

    Excludes ``"."`` (current directory) which would resolve to ``True`` for
    ``os.path.isdir`` but is almost never a valid model path.  Returns an
    empty dict for HuggingFace Hub IDs and relative names that do not exist.

    Newer huggingface_hub rejects bare filesystem paths unless
    ``local_files_only=True`` is passed.  This helper avoids
    ``HFValidationError`` for models/tokenizers stored on disk.
    """
    if path and path != "." and (os.path.isdir(path) or os.path.isfile(path)):
        return {"local_files_only": True}
    return {}

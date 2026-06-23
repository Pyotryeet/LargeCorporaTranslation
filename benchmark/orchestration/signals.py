"""Signal handler — aggressive immediate shutdown on SIGINT/SIGTERM.

Two-phase behavior:
  1st SIGINT  → kill the running translation, stop all threads, flush metrics.
  2nd SIGINT  → hard exit (os._exit) — no cleanup, just die.

On ANY signal, we immediately:
  - Set the shutdown event (translation loop stops).
  - Flush all pending metrics to disk.
  - Return GPU memory (torch.cuda.empty_cache() / MPS sync).
  - Exit cleanly.
"""

import atexit
import gc
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

# Module-level registry of cleanup callbacks.
_cleanup_callbacks: list[tuple[str, Callable[..., Any]]] = []
_cleanup_lock = threading.Lock()


def register_cleanup(name: str, fn: Callable[..., Any]) -> None:
    """Register a function to be called on shutdown.

    Called in registration order (FIFO), best-effort.
    """
    with _cleanup_lock:
        _cleanup_callbacks.append((name, fn))


def _run_cleanup() -> None:
    """Execute all registered cleanup callbacks, then free GPU memory.

    All callbacks MUST be synchronous file I/O operations — any async
    I/O spawned by a callback would need its own completion barrier,
    which is not yet implemented.  The final call to
    :func:`logging.shutdown` in the caller ensures log handlers are
    flushed before ``sys.exit``.
    """
    with _cleanup_lock:
        callbacks = list(_cleanup_callbacks)
        _cleanup_callbacks.clear()

    for name, fn in callbacks:
        try:
            fn()
        except Exception as e:
            logger.debug("Cleanup '%s' failed (non-fatal): %s", name, e)

    # ── Aggressively free GPU memory ──
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Save and restore the current device — iterating
            # set_device() permanently changes the active device.
            saved_device = torch.cuda.current_device()
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.set_device(saved_device)
            logger.debug("CUDA memory cleared across all %d device(s)", torch.cuda.device_count())
    except (RuntimeError, torch.cuda.CudaError, torch.cuda.OutOfMemoryError):
        pass

    # MPS: there's no explicit free, but sync + gc helps.
    if torch.backends.mps.is_available():
        try:
            # Force a synchronization point.
            torch.mps.synchronize()
        except (RuntimeError, OSError):
            pass

    # Force Python garbage collection to release tensor references.
    gc.collect()
    if hasattr(gc, 'garbage') and gc.garbage:
        gc.garbage.clear()


class SignalHandler:
    """Aggressive signal handler — kills the run, frees memory, exits cleanly.

    First SIGINT/SIGTERM  → stop translation loop, flush metrics, free GPU.
    Second SIGINT/SIGTERM  → os._exit(1) (immediate, no cleanup).

    Usage
    -----
    >>> signals = SignalHandler()
    >>> while not signals.killed:
    ...     do_work()
    >>> signals.cleanup()  # called in finally, safe to call multiple times.
    """

    # Track the latest instance so atexit only fires once.
    _current_instance: "SignalHandler | None" = None

    def __init__(self) -> None:
        # If a previous instance exists, restore it before installing ours.
        if SignalHandler._current_instance is not None:
            try:
                SignalHandler._current_instance.restore()
            except Exception:
                pass
        SignalHandler._current_instance = self

        self.killed = threading.Event()
        self._needs_cleanup = False
        self._signal_count = 0
        self.signal_number: int | None = None

        # Save originals for restore.
        self._orig_sigint = signal.getsignal(signal.SIGINT)
        self._orig_sigterm = signal.getsignal(signal.SIGTERM)

        # Install our handler.
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

        # Register atexit as a final safety net.
        atexit.register(self._atexit_cleanup)

        logger.debug("Signal handler installed (SIGINT/SIGTERM → kill + free memory)")

    def _handle(self, signum: int, frame: object) -> None:
        """Signal handler — must NEVER block. Only sets threading.Event flags.

        All cleanup (GPU sync, I/O, sleep, sys.exit) is deferred to the main
        loop which polls self.killed and calls cleanup() when it sees the flag.

        IMPORTANT: This handler does NOT acquire any locks.  CPython signal
        handlers execute on the main thread, and acquiring a non-reentrant
        lock that the interrupted code already holds causes an instant
        deadlock.  Simple integer arithmetic is GIL-protected and safe.
        """
        self._signal_count += 1
        count = self._signal_count
        self.signal_number = signum

        if count == 1:
            self.killed.set()
            self._needs_cleanup = True

        elif count >= 2:
            # Second signal — hard kill. No cleanup, no flush, just die.
            os._exit(128 + signum)

    def _atexit_cleanup(self) -> None:
        """Safety net: if the process exits without cleanup, free GPU here."""
        try:
            _run_cleanup()
        except Exception:
            logger.exception("atexit cleanup failed — data may be lost")

    def cleanup(self) -> None:
        """Manually trigger cleanup. Called from the main loop when self.killed is set.

        Idempotent — safe to call multiple times.
        """
        self.killed.set()
        if self._needs_cleanup:
            self._needs_cleanup = False
            sig_name = signal.Signals(self.signal_number).name if self.signal_number else "UNKNOWN"
            logger.warning(
                "Received %s — killing run, flushing metrics, freeing GPU memory...",
                sig_name,
            )
            _run_cleanup()
            # Flush all logging handlers before exiting.
            logging.shutdown()
            sys.exit(128 + (self.signal_number or signal.SIGTERM))

    def restore(self) -> None:
        """Restore original signal handlers."""
        signal.signal(signal.SIGINT, self._orig_sigint)
        signal.signal(signal.SIGTERM, self._orig_sigterm)
        atexit.unregister(self._atexit_cleanup)

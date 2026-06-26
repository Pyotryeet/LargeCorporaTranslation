#!/usr/bin/env python3
"""Surgically remove dead optimization stubs from autoregressive.py.
One-pass, content-based deletion. Idempotent — safe to run multiple times."""

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / 'benchmark' / 'inference' / 'backends' / 'autoregressive.py'

with open(TARGET, 'r') as f:
    lines = f.readlines()

print(f"Original: {len(lines)} lines")

kept = []
i = 0

while i < len(lines):
    line = lines[i]

    # --- Pattern 1: Module-level dead sentinels ---
    if any(line.startswith(s) for s in [
        'triton = None',
        'CUDAGraphDecoder = None',
        'CUDAGraphPool = None',
        '_fused_rms_norm_fn = None',
        '_fused_swiglu_fn = None',
    ]):
        i += 1; continue

    # --- Pattern 2: Lazy import try/except blocks (skip try+import+except+pass = 4 lines) ---
    if any(kw in line for kw in [
        'import triton  # noqa',
        'from benchmark.hardware.cuda_graphs import',
        'from benchmark.hardware.fused_ops import',
    ]):
        i += 4; continue

    # Lines setting variables from dead imports
    if '_fused_rms_norm_fn = fused_rms_norm_residual' in line:
        i += 1; continue
    if '_fused_swiglu_fn = fused_swiglu_gate_up' in line:
        i += 1; continue

    # --- Pattern 3: __init__ dead attributes ---
    if 'self._use_cuda_graph: bool = False' in line:
        i += 1; continue
    if 'self._use_fused_kernels' in line and 'extra.get' in line:
        i += 1; continue
    if line.strip() in ('self._use_cuda_graph = False', 'self._use_fused_kernels = False'):
        i += 1; continue
    if any(kw in line for kw in [
        'self._graph_decoder: Optional["CUDAGraphDecoder"]',
        'self._graph_pool: Optional["CUDAGraphPool"]',
        'self._fused_rms_norm: Any = None',
        'self._fused_swiglu: Any = None',
    ]):
        i += 1; continue

    # --- Pattern 4: if False block ---
    if 'if False:  # was: self._use_fused_kernels' in line:
        i += 2; continue  # skip the if + the call line

    # --- Pattern 5: _inject_fused_kernels method ---
    if 'def _inject_fused_kernels(self)' in line:
        base_indent = len(line) - len(line.lstrip())
        i += 1
        while i < len(lines):
            sl = lines[i].lstrip()
            cur_indent = len(lines[i]) - len(sl)
            if sl.startswith('def ') and cur_indent == base_indent:
                break
            i += 1
        continue

    # --- Pattern 6: CUDA GRAPH CAPTURE section + _capture_decode_graph ---
    if '# CUDA GRAPH CAPTURE' in line:
        i += 1
        while i < len(lines):
            sl = lines[i].lstrip()
            if sl.startswith('def ') and 'warmup' in sl:
                break
            i += 1
        continue

    # --- Pattern 7: Phase 3 warmup ---
    if 'Phase 3: CUDA graph capture (EXTREME)' in line:
        i += 1
        while i < len(lines):
            if 'self._capture_decode_graph(' in lines[i]:
                i += 1
                while i < len(lines):
                    if 'self._graph_decoder = None' in lines[i]:
                        i += 1
                        break
                    i += 1
                break
            i += 1
        continue

    # --- Pattern 8: close() cleanup for graph ---
    if any(kw in line for kw in [
        'if self._graph_decoder is not None:',
        'if self._graph_pool is not None:',
    ]):
        i += 1
        if i < len(lines) and 'self._graph_decoder = None' in lines[i]:
            i += 1
            if i < len(lines) and '_already_freed' in lines[i]:
                i += 1
        continue

    # --- Keep this line ---
    kept.append(line)
    i += 1

# Write back
with open(TARGET, 'w') as f:
    f.writelines(kept)

print(f"Kept: {len(kept)} lines")
print(f"Removed: {len(lines) - len(kept)} lines")

# Quick syntax check
import ast
try:
    ast.parse(''.join(kept))
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")
    sys.exit(1)

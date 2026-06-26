#!/usr/bin/env python3
"""Remove jit_compiler module and all its references from the codebase."""

import ast, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 1. Delete the module ──
mod = ROOT / 'benchmark' / 'hardware' / 'jit_compiler.py'
if mod.exists():
    mod.unlink()
    print(f'Deleted: {mod.relative_to(ROOT)}')

# ── 2. Clean autoregressive.py ──
ar = ROOT / 'benchmark' / 'inference' / 'backends' / 'autoregressive.py'
with open(ar) as f:
    lines = f.readlines()

kept = []
i = 0; removed = 0
while i < len(lines):
    line = lines[i]
    # Sentinel
    if line.startswith('precompile_all_kernels = None'):
        removed += 1; i += 1; continue
    # Lazy import block
    if 'from benchmark.hardware.jit_compiler import' in line:
        i += 4; removed += 4; continue  # try, import, except, pass
    # The call site
    if 'n_jit = precompile_all_kernels()' in line:
        removed += 1; i += 1; continue
    # JIT-related comments/log lines
    if 'JIT kernels unavailable' in line or 'JIT CUDA kernels' in line:
        removed += 1; i += 1; continue
    if '_jit_kernels_active' in line:
        removed += 1; i += 1; continue
    if 'JIT kernel precompile' in line or 'precompile_all_kernels' in line:
        removed += 1; i += 1; continue
    kept.append(line); i += 1

with open(ar, 'w') as f:
    f.writelines(kept)
try:
    ast.parse(''.join(kept))
except SyntaxError as e:
    print(f'Syntax ERROR: {e}'); sys.exit(1)
print(f'autoregressive.py: {len(lines)} → {len(kept)} lines (-{removed})')

# ── 3. Clean hardware/__init__.py ──
hw_init = ROOT / 'benchmark' / 'hardware' / '__init__.py'
with open(hw_init) as f:
    content = f.read()
# Remove the jit_compiler import block
old = '''from benchmark.hardware.jit_compiler import (
    JITCompiler,
    get_jit_compiler,
    precompile_all_kernels,
    get_kernel,
)
'''
content = content.replace(old, '')
# Remove the __all__ entries
for name in ('"JITCompiler",', '"get_jit_compiler",', '"precompile_all_kernels",', '"get_kernel",'):
    content = content.replace(f'    {name}\n', '')
# Update docstring
content = content.replace('v3.3 additions: Runtime JIT kernel compilation (CUDA C++ / PTX / Metal MSL),\n                Triton fused kernels, CUDA graphs, KV-cache quantization.', '')
with open(hw_init, 'w') as f:
    f.write(content)
try:
    ast.parse(content)
    print('hardware/__init__.py: OK')
except SyntaxError as e:
    print(f'Syntax ERROR in __init__: {e}')

# ── 4. Final stale check ──
for kw in ('jit_compiler', 'JITCompiler', 'precompile_all_kernels', 'get_jit_compiler'):
    for fpath in [ar, hw_init]:
        with open(fpath) as f:
            if kw in f.read():
                print(f'WARNING: "{kw}" still in {fpath.relative_to(ROOT)}')
                sys.exit(1)
print('Zero stale references. Done.')

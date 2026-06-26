#!/usr/bin/env python3
"""Remove kv_cache_quant module and all its references from autoregressive.py.
Content-based matching — safe to run multiple times."""

import ast, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 1. Delete the module ──
mod = ROOT / 'benchmark' / 'hardware' / 'kv_cache_quant.py'
if mod.exists():
    mod.unlink()
    print(f'Deleted: {mod.relative_to(ROOT)}')

# ── 2. Clean autoregressive.py ──
ar = ROOT / 'benchmark' / 'inference' / 'backends' / 'autoregressive.py'
with open(ar) as f:
    lines = f.readlines()

kept = []
i = 0
removed = 0

while i < len(lines):
    line = lines[i]

    # Dead sentinels
    if line.startswith('KVQuantConfig = None') or line.startswith('QuantizedKVCache = None'):
        removed += 1; i += 1; continue

    # Lazy import block for kv_cache_quant
    if 'from benchmark.hardware.kv_cache_quant import' in line:
        removed += 4; i += 4; continue  # try, import, except, pass

    # The call site in load()
    if 'self._init_kv_cache_quantization()' in line:
        removed += 1; i += 1; continue

    # _init_kv_cache_quantization method — skip from def to next top-level def
    if 'def _init_kv_cache_quantization(self)' in line:
        base = len(line) - len(line.lstrip())
        i += 1; removed += 1
        while i < len(lines):
            sl = lines[i].lstrip()
            ci = len(lines[i]) - len(sl)
            if sl.startswith('def ') and ci == base:
                break
            removed += 1; i += 1
        continue

    # _kv_quant attributes in init
    if '_kv_quant_config' in line or '_kv_quant_cache' in line:
        removed += 1; i += 1; continue

    # _kv_quant_config/cache references in close/cleanup
    if 'self._kv_quant_cache is not None' in line or \
       'self._kv_quant_cache = None' in line or \
       'self._kv_quant_config = None' in line:
        removed += 1; i += 1; continue

    kept.append(line)
    i += 1

with open(ar, 'w') as f:
    f.writelines(kept)

# Verify
try:
    ast.parse(''.join(kept))
    print(f'Syntax: OK')
except SyntaxError as e:
    print(f'Syntax ERROR: {e}')
    sys.exit(1)

print(f'autoregressive.py: {len(lines)} → {len(kept)} lines (-{removed})')

# ── 3. Strip triple blanks ──
with open(ar) as f:
    content = f.read()
while '\n\n\n\n' in content:
    content = content.replace('\n\n\n\n', '\n\n\n')
with open(ar, 'w') as f:
    f.write(content)

# ── 4. Final stale check ──
with open(ar) as f:
    final = f.read()
for kw in ('kv_cache_quant', 'KVQuantConfig', 'QuantizedKVCache', '_kv_quant'):
    if kw in final:
        print(f'WARNING: stale ref "{kw}" still present!')
        sys.exit(1)

print('Zero stale references.')
print('Done.')

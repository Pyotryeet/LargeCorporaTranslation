#!/bin/bash
# NGC Container — install deps + benchmark
set -e

# Install compatible dependency versions
pip install --no-cache-dir -q "transformers>=4.51,<5.0" orjson pyarrow safetensors sacrebleu pydantic pyyaml psutil

python3 -c 'import transformers; print("transformers:", transformers.__version__)'
python3 -c 'import orjson; print("orjson: OK")'

python3 /bench.py

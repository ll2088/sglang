#!/usr/bin/env bash
# Install only the dependencies needed to run the quantization regression
# tests in a CPU-only CI environment. Avoids `pip install -e python`, which
# pulls sglang's full runtime dependency set (py-spy, quack-kernels, ...).
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -c constraints-ci.txt pytest numpy torch

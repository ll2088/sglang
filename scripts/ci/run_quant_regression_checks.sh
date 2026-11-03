#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the quant regression checks used by the port validation plan.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TESTS=(
  test/registered/quant/test_quant_utils_defaults.py
  test/registered/quant/test_dataclass_optional_fields.py
  test/registered/quant/test_pickle_loading_safety.py
  test/registered/quant/test_moe_init_keywords.py
)

echo "== static pattern checks =="
bash "${SCRIPT_DIR}/check_no_mutable_defaults.sh"

echo "== pytest (single run) =="
python -m pytest -q "${TESTS[@]}"

echo "== pytest (stability, 3 runs, no random order) =="
for i in 1 2 3; do
  echo "-- run ${i} --"
  python -m pytest -q -p no:randomly "${TESTS[@]}"
done

echo "== import checks =="
python -c "import sglang.srt.layers.quantization.fp8, sglang.srt.layers.quantization.utils, sglang.srt.layers.quantization.compressed_tensors.utils, sglang.srt.layers.quantization.quark.utils, sglang.srt.models.inkling_common.quantization.config, sglang.srt.utils.hf_transformers.config, sglang.srt.model_loader.utils"

echo "All quant regression checks passed."

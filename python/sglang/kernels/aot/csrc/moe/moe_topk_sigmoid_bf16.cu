// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 SGLang Team. All Rights Reserved.
//
// Fused sigmoid topk launcher: gating_output=bfloat16. Including the kernel
// unit with SGLANG_TOPK_SIGMOID_IMPL set emits only this dtype's device code,
// so the per-dtype instantiations are generated in parallel translation units.
#define SGLANG_TOPK_SIGMOID_IMPL 1
#include "moe_topk_sigmoid_kernels.cu"

SGLANG_TOPK_SIGMOID_LAUNCHER_INSTANTIATE(, __nv_bfloat16);

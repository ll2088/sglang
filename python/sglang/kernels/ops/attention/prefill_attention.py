# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()

if _is_cuda or _is_hip:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()


def _on_gfx1151() -> bool:
    """Whether the split-D head-dim blocking below was tuned on this device.

    Kept local so the kernel module has no import-time dependency on the ROCm
    platform helpers; any failure to identify the device falls back to False.
    """
    if not _is_hip:
        return False
    try:
        from sglang.srt.utils import get_device_arch

        arch = get_device_arch()
    except Exception:
        return False
    return arch is not None and "gfx1151" in str(arch)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    HEAD_STRIDE_ALIGNED_8: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Triton only marks stride_*h divisible by 8 when head_dim is a multiple of
    # 16, so at head_dim=72 the D-contiguous Q/K/V loads lower to scalar ones.
    # The hint is keyed off the real strides and is a no-op where the compiler
    # already knew the alignment.
    off_h_q = cur_head * stride_qh
    off_h_k = cur_kv_head * stride_kh
    off_h_v = cur_kv_head * stride_vh
    off_h_o = cur_head * stride_oh
    if HEAD_STRIDE_ALIGNED_8:
        off_h_q = tl.multiple_of(off_h_q, 8)
        off_h_k = tl.multiple_of(off_h_k, 8)
        off_h_v = tl.multiple_of(off_h_v, 8)
        off_h_o = tl.multiple_of(off_h_o, 8)

    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + off_h_q
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + off_h_k + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + off_h_v + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Split-D path: a non-power-of-2 head_dim (e.g. 72) would otherwise force
    # BLOCK_DMODEL = next_pow2(Lk) = 128, so the qk/pv dots run 8 K-passes when
    # only ceil(Lk / 16) are needed. Covering D as a power-of-2 main block plus
    # a power-of-2 tail block (72 -> 64 + 16 = 80) drops that to 5. Constexpr
    # guarded: BLOCK_DMODEL_TAIL == 0 is dead code, so every power-of-2 head_dim
    # compiles to the single-block kernel exactly as before.
    if BLOCK_DMODEL_TAIL > 0:
        offs_dt = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL_TAIL)
        mask_dt = offs_dt < Lk
        off_qt = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
            + off_h_q
            + offs_dt[None, :]
        )
        off_kt = offs_n[None, :] * stride_kbs + off_h_k + offs_dt[:, None]
        off_vt = offs_n[:, None] * stride_vbs + off_h_v + offs_dt[None, :]
        qt = tl.load(
            Q + off_qt,
            mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_dt[None, :]),
            other=0.0,
        )
        kt_ptrs = K + off_kt
        vt_ptrs = V + off_vt

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if BLOCK_DMODEL_TAIL > 0:
        acc_tail = tl.zeros([BLOCK_M, BLOCK_DMODEL_TAIL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    end_n = (
        cur_batch_seq_len
        if not IS_CAUSAL
        else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    )
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL_TAIL > 0:
            kt = tl.load(
                kt_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                mask=((start_n + offs_n[None, :]) < cur_batch_seq_len)
                & (mask_dt[:, None]),
                other=0.0,
            )
            qk += tl.dot(qt, kt)
        qk *= sm_scale

        if IS_CAUSAL:
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_seq_len)
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),
                0,
                float("-inf"),
            )
        else:
            qk += tl.where(
                (start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float("-inf")
            )

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        if BLOCK_DMODEL_TAIL > 0:
            vt = tl.load(
                vt_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                mask=((start_n + offs_n[:, None]) < cur_batch_seq_len)
                & (mask_dt[None, :]),
                other=0.0,
            )
            acc_tail += tl.dot(p, vt)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + off_h_o
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )
    if BLOCK_DMODEL_TAIL > 0:
        off_ot = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
            + off_h_o
            + offs_dt[None, :]
        )
        tl.store(
            Out + off_ot,
            acc_tail,
            mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_dt[None, :]),
        )


def context_attention_fwd(
    q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True, sm_scale=None
):
    """
    q, k, v: [b * s, head, head_dim]
    b_start_loc: [b]
    b_seq_len: [b]
    out: [b * s, head, head_dim]
    sm_scale: softmax scale, defaults to 1/sqrt(head_dim)
    """
    if (_is_cuda or _is_hip) and CUDA_CAPABILITY[0] > 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DMODEL_TAIL = 0
    HEAD_STRIDE_ALIGNED_8 = False

    if _on_gfx1151() and q.dtype in (torch.bfloat16, torch.float16):
        main = BLOCK_DMODEL // 2
        tail = max(16, triton.next_power_of_2(Lk - main))
        if Lk != BLOCK_DMODEL and main + tail < BLOCK_DMODEL:
            BLOCK_DMODEL, BLOCK_DMODEL_TAIL = main, tail

        # The hint is only valid when every head-axis stride is 8-element
        # aligned; derive it from the real strides so padded views fall back.
        HEAD_STRIDE_ALIGNED_8 = all(
            t.stride(1) % 8 == 0 for t in (q, k, v, o)
        )

        if HEAD_STRIDE_ALIGNED_8 and (BLOCK_DMODEL, BLOCK_DMODEL_TAIL) == (64, 16):
            BLOCK_N = 16
            num_warps = 4
        else:
            BLOCK_N = BLOCK
    else:
        BLOCK_N = BLOCK

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DMODEL_TAIL=BLOCK_DMODEL_TAIL,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
        HEAD_STRIDE_ALIGNED_8=HEAD_STRIDE_ALIGNED_8,
    )

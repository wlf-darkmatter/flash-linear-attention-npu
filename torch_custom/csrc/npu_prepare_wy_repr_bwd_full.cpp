/**
 * @file npu_prepare_wy_repr_bwd_full.cpp
 *
 * Copyright (C) 2024-2025.Tianjin University, Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include "npu_cpp_extension.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using variable_list = std::vector<at::Tensor>;

// 为NPU设备注册前向实现
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_prepare_wy_repr_bwd_full(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    const at::Tensor &dA,
    const at::Tensor &dw,
    const at::Tensor &du,
    const at::Tensor &g,
    const c10::optional<at::IntArrayRef>& cu_seqlens,
    const c10::optional<at::IntArrayRef>& chunk_indices,
    int64_t chunk_size)
{
    // 创建输出内存
    at::Tensor dk = at::empty_like(k);
    at::Tensor dv = at::empty_like(v);
    at::Tensor dg = at::empty_like(g);
    at::Tensor dbeta =  at::empty_like(beta);

    // 调用aclnn接口计算
    EXEC_NPU_CMD_EXT(aclnnPrepareWyReprBwdFull, k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, chunk_size, dk, dv, dbeta, dg);
    return std::tie(dk, dv, dbeta, dg);
}

// 为NPU设备注册实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("npu_prepare_wy_repr_bwd_full", torch::dispatch(
               c10::DispatchKey::PrivateUse1,
               [](const at::Tensor& k, 
                  const at::Tensor& v,
                  const at::Tensor& beta,
                  const at::Tensor& a,
                  const at::Tensor& dA,
                  const at::Tensor& dW,
                  const at::Tensor& du,
                  const at::Tensor& g,
                  const c10::optional<at::IntArrayRef>& cu_seqlens,
                  const c10::optional<at::IntArrayRef>& chunk_indices,
                  int64_t chunk_size) {
                   return npu_prepare_wy_repr_bwd_full(
                       k, v, beta, a, dA, dW, du, g,
                       cu_seqlens, chunk_indices, chunk_size);
               }));
}
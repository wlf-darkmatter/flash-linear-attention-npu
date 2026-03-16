/**
 * @file function.h
 *
 * Copyright (C) 2024-2025. Tianjin University, Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef FUNCTION_H
#define FUNCTION_H

#include <ATen/ATen.h>

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
    int64_t chunk_size);



#endif //  FUNCTION_H
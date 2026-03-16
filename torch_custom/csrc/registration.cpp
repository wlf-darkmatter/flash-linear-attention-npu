/**
 * @file registration.cpp
 *
 * Copyright (C) 2024-2025. Tianjin University, Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <torch/extension.h>
#include "function.h"

// 在myops命名空间里注册add_custom和add_custom_backward两个schema，新增自定义aten ir需要在此注册
TORCH_LIBRARY(myops, m) {
    m.def("npu_prepare_wy_repr_bwd_full(Tensor k, Tensor v, Tensor beta, Tensor a, Tensor dA, Tensor dW, Tensor du, Tensor g, int[]? cu_seqlens, int[]? chunk_indices, int chunk_size) -> (Tensor, Tensor, Tensor, Tensor)");
}

// 通过pybind将c++接口和python接口绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_prepare_wy_repr_bwd_full", &npu_prepare_wy_repr_bwd_full, "Prepare WY representation backward full implementation");
}
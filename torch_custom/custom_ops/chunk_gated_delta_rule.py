import torch
import custom_ops_lib

def npu_prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, chunk_size):
    return custom_ops_lib.npu_prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, chunk_size)
import torch
import torch_npu
from typing import Optional
import math
import hashlib
from ct import single
from golden import chunk_bwd_dv_local_fix, chunk_bwd_dv_local_variable, prepare_chunk_indices
from utils import generate_cu_seqlens, compare_tensors_by_ratio, create_incremental_tensor, create_tensor, bool_matrix_to_uint8, get_tensor_md5, compare_tensors_md5

def test_variable():
    B, H, T, K, V = 1, 32, 32768, 128, 128
    chunk_size=128
    scale = 2.0
    cu_seqlens_len = 4

    q = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== q.shape = {q.shape} ")
    k = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== k.shape = {k.shape} ")
    d_o = create_tensor((B, H, T, V), dtype=torch.float16)
    print(f"==== d_o.shape = {d_o.shape} ")
    g = create_tensor((B, H, T), dtype=torch.float16)
    print(f"==== g.shape = {g.shape} ")
    # print("q =",q)
    # print("k =",k)
    # print("d_o =",d_o)
    # print("g =",g)
    upper_tri_matrix = bool_matrix_to_uint8(chunk_size)
    # upper_tri_matrix = ~upper_tri_matrix
    # print(f"==== upper_tri_matrix.shape = {upper_tri_matrix.shape} ",upper_tri_matrix)
    
    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    print(f"==== cu_seqlens.shape = {cu_seqlens.shape} ",cu_seqlens)

    # dv_golden = chunk_bwd_dv_local_variable(q, k, d_o, g, scale, cu_seqlens, chunk_size)

    # q_npu = q.npu()
    # k_npu = k.npu()
    # d_o_npu = d_o.npu()
    # g_npu = g.npu()
    # upper_tri_matrix_npu = upper_tri_matrix.npu()
    # if cu_seqlens is not None:
    #     cu_seqlens_npu = cu_seqlens.npu()
    #     chunk_indices_npu = chunk_indices.npu()

    # dv = torch_npu.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu,upper_tri_matrix=upper_tri_matrix_npu, g_gamma=None, A=None,cu_seqlens=cu_seqlens_npu, chunk_indices = chunk_indices_npu, scale=scale, chunk_size =chunk_size)
    # # torch.save(dv.cpu(),"dv_pre.pt")
    
    # # torch.save(dv.cpu(),"dv.pt")
    # # torch.save(dv_golden,"dv_golden.pt")

    # dv_golden = torch.load("dv_golden.pt")
    # compare_tensors_by_ratio(dv_golden,dv.cpu())

    # dv_pre = torch.load("dv_pre.pt")
    # md5_1, md5_2, is_equal = compare_tensors_md5(dv_pre, dv.cpu())
    # print(f"dv_pre  Shape: {dv_pre.shape}")
    # print(f"  MD5: {md5_1}")
    # print(f"dv  Shape: {dv.shape}")
    # print(f"  MD5: {md5_2}")
    # print(f"\nMD5对比结果: {'相同' if is_equal else '不同'}")

def test_fix():
    B, H, T, K, V = 256, 32, 128, 128, 128
    chunk_size=128
    scale = 12.0

    q = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== q.shape = {q.shape} ")
    k = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== k.shape = {k.shape} ")
    d_o = create_tensor((B, H, T, V), dtype=torch.float16)
    print(f"==== d_o.shape = {d_o.shape} ")
    g = create_tensor((B, H, T), dtype=torch.float16)
    print(f"==== g.shape = {g.shape} ")
    # print("q =",q)
    # print("k =",k)
    # print("d_o =",d_o)
    # print("g =",g)
    upper_tri_matrix = bool_matrix_to_uint8(chunk_size)
    # print(f"==== upper_tri_matrix.shape = {upper_tri_matrix.shape} ")
    cu_seqlens = None
    # dv_golden =  chunk_bwd_dv_local_fix(q, k, d_o, g, scale, cu_seqlens, chunk_size)
    # print(f"==== dv_golden.shape = {dv_golden.shape} ",dv_golden)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()
    upper_tri_matrix_npu = upper_tri_matrix.npu()
    dv = torch_npu.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu,upper_tri_matrix=upper_tri_matrix_npu, g_gamma=None, A=None,cu_seqlens=None, chunk_indices = None, scale=scale, chunk_size =chunk_size)
    # torch.save(dv.cpu(),"dv.pt")
    # torch.save(dv_golden,"dv_golden.pt")
    # compare_tensors_by_ratio(dv_golden,dv.cpu())
    # single(dv.cpu(),dv_golden)

if __name__ == "__main__":
    torch.manual_seed(0)
    
    test_variable()
    # test_fix()

    

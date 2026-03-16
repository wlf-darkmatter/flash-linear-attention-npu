import copy
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


def test_prepare_wy_repr_bwd_full(
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    ktype,
    btype,
    cu_seqlens = None,
    chunk_indices = None,
    seed: int = 0,
):
    """
    生成随机输入张量，保存到文件，并调用 NPU 的 WY 表示反向算子。

    参数:
        B (int): Batch size
        H (int): Head number
        T (int): Sequence length
        K (int): Key dimension
        V (int): Value dimension
        chunk_size (int): Chunk size (通常为 T 的因数)
        seed (int): 随机种子，默认为 0
        save_path (str): 保存 pickle 文件的路径，默认为 'data.pkl'

    返回:
        tuple: (dk, dv, dbeta, dg) —— 反向传播的梯度结果（在 NPU 上）
    """
    torch.manual_seed(seed)
    if not hasattr(test_prepare_wy_repr_bwd_full, "call_count"):
        test_prepare_wy_repr_bwd_full.call_count = 1
    else:
        test_prepare_wy_repr_bwd_full.call_count += 1

    # 生成随机张量（float16）
    k = torch.rand(B, H, T, K, dtype=ktype)
    v = torch.rand(B, H, T, V, dtype=ktype)
    beta = torch.rand(B, H, T, dtype=btype)
    A = torch.rand(B, H, T, chunk_size, dtype=ktype)
    dA = torch.rand(B, H, T, chunk_size, dtype=ktype)
    dw = torch.rand(B, H, T, K, dtype=ktype)
    du = torch.rand(B, H, T, V, dtype=ktype)
    g = torch.rand(B, H, T, dtype=btype)

    # 将张量移到 NPU 并调用反向算子
    if chunk_indices != None:
        dk, dv, dbeta, dg = custom_ops.npu_prepare_wy_repr_bwd_full(
            k.npu(),
            v.npu(),
            beta.npu(),
            A.npu(),
            dA.npu(),
            dw.npu(),
            du.npu(),
            g.npu(),
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size
        )
    else:
        dk, dv, dbeta, dg = custom_ops.npu_prepare_wy_repr_bwd_full(
            k.npu(),
            v.npu(),
            beta.npu(),
            A.npu(),
            dA.npu(),
            dw.npu(),
            du.npu(),
            g.npu(),
            cu_seqlens=None,
            chunk_indices=None,
            chunk_size=chunk_size
        )
    if chunk_indices!=None:
        NT = len(chunk_indices) // 2
    else:
        NT = (T + chunk_size - 1) // chunk_size
    # cpu_dv = compute_dv_golden(A, du, beta, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
    # ct.isclose(dv, cpu_dv, diff_thd=0.1)
    # # torch.save(cpu_dv, "cpu_dv.pt")
    
    # cpu_dk = compute_dk_golden(A, dw, g, beta, dA,k, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
    # ct.isclose(dk, cpu_dk, diff_thd=0.1)
    # # torch.save(cpu_dk, "cpu_dk.pt")
    
    # cpu_dg = compute_dg_golden(A, dw, g, beta, dA,k, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
    # ct.isclose(dg, cpu_dg, diff_thd=0.1)
    # # torch.save(cpu_dg, "cpu_dg.pt")
    
    # cpu_dbeta = compute_dbeta_golden(A, dw, g, beta, dA,k,v,du, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
    # ct.isclose(dbeta, cpu_dbeta, diff_thd=0.1)
    # # torch.save(cpu_dbeta, "cpu_dbeta.pt") 
    
    print(f"test_prepare_wy_repr_bwd_full 被调用了第 {test_prepare_wy_repr_bwd_full.call_count} 次")
    return dk, dv, dbeta, dg

if __name__ == "__main__":
    #F1
    test_prepare_wy_repr_bwd_full(B = 64, H = 8, T = 1024, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16)
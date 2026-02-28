/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h.cpp
 * \brief
 */

// #include "chunk_gated_delta_rule_fwd_h.h"
#include "catlass/gemm/kernel/gdn_fwd_h_kernel.hpp"
#include "lib/matmul_intf.h"

using namespace Catlass;

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g,
                                                         GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                         GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    
    __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);
    if (gdnFwdHTilingData->dataType == 0) {

        using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHCube;
        using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHVec;
        
        using DispatchPolicy = Gemm::MmadAtlasA2PingpongGdn<true>;
        using L1TileShape = GemmShape<128, 128, 128>;
        using L0TileShape = L1TileShape;

        using WType = Gemm::GemmType<half, layout::RowMajor>;
        using HType = Gemm::GemmType<half, layout::RowMajor>;
        using Vworkype = Gemm::GemmType<half, layout::RowMajor>;
        using KType = Gemm::GemmType<half, layout::ColumnMajor>;
        using HworkType = Gemm::GemmType<half, layout::RowMajor>;
        using VType = Gemm::GemmType<half, layout::RowMajor>;
        using GType = Gemm::GemmType<float, layout::RowMajor>;
        using UType = Gemm::GemmType<half, layout::RowMajor>;
        using HType = Gemm::GemmType<half, layout::RowMajor>;

        // cube 1
        using BlockMmadWH = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, WType, HType, Vworkype>;

        // cube 2
        using BlockMmadKV = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, KType, VType, HworkType>;

        // vec 1
        using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasA2GDNFwdHVnew;
        using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, Vworkype>;

        // vec 2
        using DispatchPolicyGDNFwdHUpdate = Epilogue::EpilogueAtlasA2GDNFwdHUpdate;
        using EpilogueGDNFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHUpdate, HType, GType, HType, HworkType>;

        using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<CubeScheduler, VecScheduler, BlockMmadWH, BlockMmadKV, EpilogueGDNFwdHVnew, EpilogueGDNFwdHUpdate>;

        GDNFwdHKernel gdnFwdH;
        gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
        gdnFwdH.Process();

    } else {
        
        using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHCube;
        using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHVec;
        
        using DispatchPolicy = Gemm::MmadAtlasA2PingpongGdn<true>;
        using L1TileShape = GemmShape<128, 128, 128>;
        using L0TileShape = L1TileShape;

        using WType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
        using HType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
        using Vworkype = Gemm::GemmType<half, layout::RowMajor>;
        using KType = Gemm::GemmType<bfloat16_t, layout::ColumnMajor>;
        using HworkType = Gemm::GemmType<half, layout::RowMajor>;
        using VType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
        using GType = Gemm::GemmType<float, layout::RowMajor>;
        using UType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
        using HType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;

        // cube 1
        using BlockMmadWH = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, WType, HType, Vworkype>;

        // cube 2
        using BlockMmadKV = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, KType, VType, HworkType>;

        // vec 1
        using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasA2GDNFwdHVnew;
        using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, Vworkype>;

        // vec 2
        using DispatchPolicyGDNFwdHUpdate = Epilogue::EpilogueAtlasA2GDNFwdHUpdate;
        using EpilogueGDNFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHUpdate, HType, GType, HType, HworkType>;

        using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<CubeScheduler, VecScheduler, BlockMmadWH, BlockMmadKV, EpilogueGDNFwdHVnew, EpilogueGDNFwdHUpdate>;

        GDNFwdHKernel gdnFwdH;
        gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
        gdnFwdH.Process();
    }
}

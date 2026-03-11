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
 * \file chunk_fwd_o.cpp
 * \brief
 */

// #include "chunk_fwd_o.h"
#include "catlass/gemm/kernel/gdn_fwd_o_kernel.hpp"
#include "lib/matmul_intf.h"

using namespace Catlass;

extern "C" __global__ __aicore__ void chunk_fwd_o(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                                                         GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_offsets,
                                                         GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    
    __gm__ ChunkFwdOTilingData *__restrict gdnFwdOTilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);
    if (gdnFwdOTilingData->dataType == 0) {
 
        using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube;
        using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOVec;
        
        using L1TileShape = GemmShape<128, 128, 128>;
        using L0TileShape = L1TileShape;
        using DispatchPolicy = Gemm::MmadAtlasA2PingpongGdn<true>;
        using QType = Gemm::GemmType<half, layout::RowMajor>;
        using KType = Gemm::GemmType<half, layout::ColumnMajor>;
        using AttenType = Gemm::GemmType<half, layout::RowMajor>;
        using AttenMaskedType = Gemm::GemmType<half, layout::RowMajor>;
        using HType = Gemm::GemmType<half, layout::RowMajor>;
        using OinterType = Gemm::GemmType<half, layout::RowMajor>;
        using VNEWType = Gemm::GemmType<half, layout::RowMajor>;

        using GType = Gemm::GemmType<float, layout::RowMajor>;
        using OType = Gemm::GemmType<half, layout::RowMajor>;
        using MaskType = Gemm::GemmType<bool, layout::RowMajor>;

        // cube 1
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, QType, KType, AttenType>;

        // cube 2
        using BlockMmadQH = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, QType, HType, OinterType>;

        // cube 3
        using BlockMmadAttenVNEW = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AttenMaskedType, VNEWType, OinterType>;

        // vec 1
        using DispatchPolicyGDNFwdOQkmask = Epilogue::EpilogueAtlasA2GDNFwdOQkmask;
        using EpilogueGDNFwdOQkmask = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOQkmask, AttenMaskedType, GType, AttenType, MaskType>;

        // vec 2
        using DispatchPolicyGDNFwdOOutput = Epilogue::EpilogueAtlasA2GDNFwdOOutput;
        using EpilogueGDNFwdOOutput = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOOutput, OType, GType, OinterType, OinterType>;

        using GDNFwdOKernel = Catlass::Gemm::Kernel::GDNFwdOKernel<CubeScheduler, VecScheduler, BlockMmadQK, BlockMmadQH, BlockMmadAttenVNEW, EpilogueGDNFwdOQkmask, EpilogueGDNFwdOOutput>;

        GDNFwdOKernel gdnFwdO;
        gdnFwdO.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
        gdnFwdO.Process();

    } else {

        using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube;
        using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOVec;
        
        using L1TileShape = GemmShape<128, 128, 128>;
        using L0TileShape = L1TileShape;
        using DispatchPolicy = Gemm::MmadAtlasA2PingpongGdn<true>;
        using QType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
        using KType = Gemm::GemmType<bfloat16_t, layout::ColumnMajor>;
        using AttenType = Gemm::GemmType<half, layout::RowMajor>;
        using AttenMaskedType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
        using HType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
        using OinterType = Gemm::GemmType<half, layout::RowMajor>;
        using VNEWType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;

        using GType = Gemm::GemmType<float, layout::RowMajor>;
        using OType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
        using MaskType = Gemm::GemmType<bool, layout::RowMajor>;

        // cube 1
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, QType, KType, AttenType>;

        // cube 2
        using BlockMmadQH = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, QType, HType, OinterType>;

        // cube 3
        using BlockMmadAttenVNEW = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AttenMaskedType, VNEWType, OinterType>;

        // vec 1
        using DispatchPolicyGDNFwdOQkmask = Epilogue::EpilogueAtlasA2GDNFwdOQkmask;
        using EpilogueGDNFwdOQkmask = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOQkmask, AttenMaskedType, GType, AttenType, MaskType>;

        // vec 2
        using DispatchPolicyGDNFwdOOutput = Epilogue::EpilogueAtlasA2GDNFwdOOutput;
        using EpilogueGDNFwdOOutput = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOOutput, OType, GType, OinterType, OinterType>;

        using GDNFwdOKernel = Catlass::Gemm::Kernel::GDNFwdOKernel<CubeScheduler, VecScheduler, BlockMmadQK, BlockMmadQH, BlockMmadAttenVNEW, EpilogueGDNFwdOQkmask, EpilogueGDNFwdOOutput>;

        GDNFwdOKernel gdnFwdO;
        gdnFwdO.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
        gdnFwdO.Process();

    }
}

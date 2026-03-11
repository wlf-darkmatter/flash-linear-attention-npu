/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_scheduler_gdn_fwd_o.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"

#include "kernel_operator.h"
using namespace Catlass;

// template <>
namespace Catlass::Gemm::Kernel {

template<
    class CubeScheduler, 
    class VecScheduler, 
    class BlockMmadQK,
    class BlockMmadQH,
    class BlockMmadAttenVNEW,
    class EpilogueGDNFwdOQkmask,
    class EpilogueGDNFwdOOutput
>
class GDNFwdOKernel {
public:
    
    using ArchTag = Arch::AtlasA2;
    using GDNFwdOOffsets = Catlass::Gemm::Block::GDNFwdOOffsets;

    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;

    using ElementK =  typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;

    using ElementAtten = typename BlockMmadQK::ElementC;
    using LayoutAtten = typename BlockMmadQK::LayoutC;
    
    using ElementAttenMasked = typename BlockMmadQH::ElementA;
    using LayoutAttenMasked = typename BlockMmadQH::LayoutA;

    using ElementH = typename BlockMmadQH::ElementB;
    using LayoutH = typename BlockMmadQH::LayoutB;

    using ElementOinter = typename BlockMmadQH::ElementC;
    using LayoutOinter = typename BlockMmadQH::LayoutC;


    using ElementVNEW = typename BlockMmadAttenVNEW::ElementB; 
    using LayoutVNEW = typename BlockMmadAttenVNEW::LayoutB;


    using ElementA = half;
    using ElementG = float;
    using ElementMask = bool;

    using L1TileShape = typename BlockMmadQK::L1TileShape;

    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    float scale;
    uint32_t numChunks;
    uint32_t isVariedLen;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t attnWorkspaceOffset;
    uint32_t aftermaskWorkspaceOffset;
    uint32_t maskWorkspaceOffset;
    
    AscendC::GlobalTensor<ElementQ> gmQ;
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementVNEW> gmV;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementVNEW> gmO;
    AscendC::GlobalTensor<ElementOinter> gmVWorkspace;
    AscendC::GlobalTensor<ElementOinter> gmHWorkspace;
    AscendC::GlobalTensor<ElementAtten> gmAttnWorkspace;
    AscendC::GlobalTensor<ElementAttenMasked> gmAftermaskWorkspace;
    AscendC::GlobalTensor<ElementMask> gmMask;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;

    __aicore__ inline GDNFwdOKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g, 
        GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, GM_ADDR tiling, GM_ADDR user) {
        
        __gm__ ChunkFwdOTilingData *__restrict gdnFwdOTilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);

        shapeBatch = gdnFwdOTilingData->shapeBatch;
        seqlen = gdnFwdOTilingData->seqlen;
        kNumHead = gdnFwdOTilingData->kNumHead;
        vNumHead = gdnFwdOTilingData->vNumHead;
        kHeadDim = gdnFwdOTilingData->kHeadDim;
        vHeadDim = gdnFwdOTilingData->vHeadDim;
        scale = gdnFwdOTilingData->scale;
        chunkSize = gdnFwdOTilingData->chunkSize;
        isVariedLen = gdnFwdOTilingData->isVariedLen;
        tokenBatch = gdnFwdOTilingData->tokenBatch;
        vWorkspaceOffset = gdnFwdOTilingData->vWorkspaceOffset;
        hWorkspaceOffset = gdnFwdOTilingData->hWorkspaceOffset;
        attnWorkspaceOffset = gdnFwdOTilingData->attnWorkspaceOffset;
        aftermaskWorkspaceOffset = gdnFwdOTilingData->aftermaskWorkspaceOffset;
        maskWorkspaceOffset = gdnFwdOTilingData->maskWorkspaceOffset;

        gmQ.SetGlobalBuffer((__gm__ ElementQ *)q);
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmV.SetGlobalBuffer((__gm__ ElementVNEW *)v);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmO.SetGlobalBuffer((__gm__ ElementVNEW *)o);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + vWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + hWorkspaceOffset));
        gmAttnWorkspace.SetGlobalBuffer((__gm__ ElementAtten *)(user + attnWorkspaceOffset));
        gmAftermaskWorkspace.SetGlobalBuffer((__gm__ ElementAttenMasked *)(user + aftermaskWorkspaceOffset));
        gmMask.SetGlobalBuffer((__gm__ ElementMask *)(user + maskWorkspaceOffset));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_offsets, tiling);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_offsets, tiling);
        }
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadQK blockMmadQK(resource);
            BlockMmadQH blockMmadQH(resource);
            BlockMmadAttenVNEW blockMmadAttenVNEW(resource);

            LayoutQ qLayout{shapeBatch * kNumHead * seqlen, kHeadDim};  
            LayoutK  kLayout{kHeadDim, shapeBatch * kNumHead * seqlen};
            LayoutH  hLayout{shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim};
            LayoutOinter ointerLayout{coreNum * chunkSize * PING_PONG_STAGES, vHeadDim};
            LayoutVNEW  vnewLayout{shapeBatch * vNumHead * seqlen, vHeadDim};

            bool needRun = false;
            bool isFirstC3 = true;

            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();

                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {

                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);

                    GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                    int64_t cube1OffsetQ = cube1Offsets.qkOffset; 
                    int64_t cube1OffsetK = cube1Offsets.qkOffset; 
                    int64_t cube1OffsetAttn = cube1Offsets.attnWorkOffset; 
                    LayoutAtten attenLayout{coreNum * chunkSize * PING_PONG_STAGES, cube1Offsets.blockTokens}; 
                    GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                    blockMmadQK.preSetFlags();
                    blockMmadQK(gmQ[cube1OffsetQ], qLayout,
                            gmK[cube1OffsetK],  kLayout,
                            gmAttnWorkspace[cube1OffsetAttn],  attenLayout,
                            cube1Shape);
                    blockMmadQK.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done);

                }

                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum) {


                    if(!cubeBlockScheduler.isRunning) Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);

                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);

                    GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                    int64_t cube2OffsetH = cube2Offsets.hOffset;
                    int64_t cube2OffsetHWork = cube2Offsets.hvWorkOffset; 
                    GemmCoord cube2Shape{cube2Offsets.blockTokens, vHeadDim, kHeadDim};
                    blockMmadQH.preSetFlags();
                    blockMmadQH(gmQ[cube2OffsetQ], qLayout,
                            gmH[cube2OffsetH],  hLayout,
                            gmHWorkspace[cube2OffsetHWork],  ointerLayout,
                            cube2Shape);
                    blockMmadQH.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done);
                }

                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum) {

                
                    GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();

                    if(isFirstC3) Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);

                    int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset; 
                    int64_t cube3OffsetV = cube3Offsets.ovOffset; 
                    int64_t cube3OffsetVWork = cube3Offsets.hvWorkOffset; 
                    LayoutAtten attenLayout{coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens}; 
                    GemmCoord cube3Shape{cube3Offsets.blockTokens, vHeadDim, cube3Offsets.blockTokens};
                    blockMmadAttenVNEW.preSetFlags();
                    blockMmadAttenVNEW(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout,
                            gmV[cube3OffsetV],  vnewLayout,
                            gmVWorkspace[cube3OffsetVWork],  ointerLayout,
                            cube3Shape);
                    blockMmadAttenVNEW.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done);
                    isFirstC3 = false;
                }
                needRun = true;

                // AscendC::PipeBarrier<PIPE_ALL>();
            }
            if (coreIdx < coreNum) {
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);

            }
        }

        if ASCEND_IS_AIV {

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

            AscendC::LocalTensor<half> maskUbTensor = resource.ubBuf.template GetBufferByByte<half>(0);
            for(uint32_t i = 0; i < chunkSize; ++i)
            {
                for(uint32_t j = 0 ; j < chunkSize; ++j)
                {
                    if(i>=j) maskUbTensor.SetValue(i*chunkSize+j, (half)1.0);
                    else maskUbTensor.SetValue(i*chunkSize+j, (half)0.0); 
                    // maskUbTensor.SetValue(i*chunkSize+j, (half)0.0);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            bool needRun = false;

            if (coreIdx < coreNum * subBlockNum) {

                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);


            }

            while (vecBlockScheduler.isRunning) {
                vecBlockScheduler.InitTask();

                if (vecBlockScheduler.isRunning && coreIdx < coreNum * subBlockNum) {
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done);
                    GDNFwdOOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
                    int64_t vec1OffsetAttnMask = vec1Offsets.attnWorkOffset;
                    int64_t vec1OffsetG = vec1Offsets.gOffset;
                    int64_t vec1OffsetAttn = vec1Offsets.attnWorkOffset;
                    EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                    epilogueGDNFwdOQkmask(
                        gmAftermaskWorkspace[vec1OffsetAttnMask], 
                        gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                        chunkSize, vec1Offsets.blockTokens, kHeadDim, vHeadDim
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
                }

                AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum * subBlockNum) {
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done);
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube3Done);
                    GDNFwdOOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
                    int64_t vec2OffsetO = vec2Offsets.ovOffset;
                    int64_t vec2OffsetG = vec2Offsets.gOffset;
                    int64_t vec2OffsetVWork = vec2Offsets.hvWorkOffset;
                    int64_t vec2OffsetHWork = vec2Offsets.hvWorkOffset;
                    EpilogueGDNFwdOOutput epilogueGDNFwdOOutput(resource);
                    epilogueGDNFwdOOutput(
                        gmO[vec2OffsetO], 
                        gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork], 
                        scale, vec2Offsets.blockTokens, kHeadDim, vHeadDim
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
                }
                
                AscendC::PipeBarrier<PIPE_ALL>();

                needRun = true;
            }
        }
    }
    
};

}

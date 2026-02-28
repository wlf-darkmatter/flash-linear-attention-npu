/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class HOutputType_,
    class GInputType_,
    class AInputType_,
    class HInputType_
>
class BlockEpilogue <
    EpilogueAtlasA2GDNFwdOOutput,
    HOutputType_,
    GInputType_,
    AInputType_,
    HInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2GDNFwdOOutput;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;
    using HElementInput = typename HInputType_::Element;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t FLOAT_ELENUM_PER_BLK = 8;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t UB_TILE_SIZE = 16384;  // 64 * 128 * 2B
    static constexpr uint32_t UB_LINE_SIZE = 512;   // 128 * 2 * 2B
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;    // 128 * 2
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;   // 128
    static constexpr uint32_t MULTIPLIER = 2;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t BASE = 0;
        constexpr uint32_t ATTN_UB_TENSOR_OFFSET = BASE + 0 * UB_LINE_SIZE; // 16*128*2B
        constexpr uint32_t H_UB_TENSOR_OFFSET = ATTN_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
        constexpr uint32_t SRC_UB_TENSOR_OFFSET = H_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE; // 16*128*128
        constexpr uint32_t FLOAT_UB_TENSOR_OFFSET = SRC_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
        constexpr uint32_t GBRC_UB_TENSOR_OFFSET = FLOAT_UB_TENSOR_OFFSET + 64 * UB_LINE_SIZE;
        constexpr uint32_t G_UB_TENSOR_OFFSET = GBRC_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
        constexpr uint32_t G_UB_HALF_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + 1 * UB_LINE_SIZE;

        aUbTensor = resource.ubBuf.template GetBufferByByte<AElementInput>(ATTN_UB_TENSOR_OFFSET);
        hUbTensor = resource.ubBuf.template GetBufferByByte<HElementInput>(H_UB_TENSOR_OFFSET);
        srcUbTensor = resource.ubBuf.template GetBufferByByte<HElementOutput>(SRC_UB_TENSOR_OFFSET);
        floatUbTensor = resource.ubBuf.template GetBufferByByte<float>(FLOAT_UB_TENSOR_OFFSET);
        gbrcUbTensor = resource.ubBuf.template GetBufferByByte<AElementInput>(GBRC_UB_TENSOR_OFFSET);
        gUbTensor = resource.ubBuf.template GetBufferByByte<GElementInput>(G_UB_TENSOR_OFFSET);
        gUbhalfTensor = resource.ubBuf.template GetBufferByByte<AElementInput>(G_UB_HALF_TENSOR_OFFSET);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<AElementInput> attnInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        float scale,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim
        )
    {
        uint32_t mCVActual = chunkSize;
        uint32_t nCVActual = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mCVActualPerSubBlock = CeilDiv(mCVActual, subBlockNum);
        uint32_t mCVActualThisSubBlock = (subBlockIdx == 0) ? mCVActualPerSubBlock : (mCVActual - mCVActualPerSubBlock);
        uint32_t mCVOffset = subBlockIdx * mCVActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetCV = mCVOffset * nCVActual + nOffset;

        AscendC::ResetMask();
        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetCV];
        AscendC::GlobalTensor<AElementInput> attnInputThisSubBlock = attnInput[offsetCV];
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetCV];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(mCVActualThisSubBlock * nCVActual * sizeof(half)), 0, 0, 0};
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mCVActualThisSubBlock * nCVActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mCVActualThisSubBlock * nCVActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gUbTensor, gInputThisSubBlock, (mCVActual + 8 - 1) / 8 * 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(gUbhalfTensor, gUbTensor, AscendC::RoundMode::CAST_NONE, mCVActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp(gUbhalfTensor, gUbhalfTensor, mCVActual); // gubhalfUbTensor
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int i = 0; i < mCVActualThisSubBlock; ++i) {
            for (int j = 0; j < nCVActual; ++j) {
                gbrcUbTensor.SetValue(i * nCVActual + j, 
                    gUbhalfTensor.GetValue(i + ((subBlockIdx == 0) ? 0 : mCVActualPerSubBlock)));
            }
        }
        
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(hUbTensor, hUbTensor, gbrcUbTensor, mCVActualThisSubBlock * nCVActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Add(aUbTensor, aUbTensor, hUbTensor, mCVActualThisSubBlock * nCVActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(aUbTensor, aUbTensor, (half)scale, mCVActualThisSubBlock * nCVActual);
        if constexpr(!std::is_same<HElementOutput, half>::value) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(floatUbTensor, aUbTensor, AscendC::RoundMode::CAST_NONE, mCVActualThisSubBlock * nCVActual);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(srcUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, mCVActualThisSubBlock * nCVActual);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(hOutputThisSubBlock, srcUbTensor, copyParams);
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(hOutputThisSubBlock, aUbTensor, copyParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::LocalTensor<half> aUbTensor;
    AscendC::LocalTensor<half> hUbTensor;
    AscendC::LocalTensor<half> gbrcUbTensor;
    AscendC::LocalTensor<float> gUbTensor;
    AscendC::LocalTensor<half> gUbhalfTensor;
    AscendC::LocalTensor<HElementOutput> srcUbTensor;
    AscendC::LocalTensor<float> floatUbTensor;
};
}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP


#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class AOutputType_,
    class GInputType_,
    class AInputType_
>
class BlockEpilogue <
    EpilogueAtlasA2GDNFwdOQkmask,
    AOutputType_,
    GInputType_,
    AInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2GDNFwdOQkmask;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using AElementOutput = typename AOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;

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
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = ATTN_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
        constexpr uint32_t SRC_UB_TENSOR_OFFSET = MASK_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE; // 16*128*128
        constexpr uint32_t FLOAT_UB_TENSOR_OFFSET = SRC_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
        constexpr uint32_t GBRCLEFT_UB_TENSOR_OFFSET = FLOAT_UB_TENSOR_OFFSET + 64 * UB_LINE_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_OFFSET = GBRCLEFT_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
        constexpr uint32_t G_UB_TENSOR_OFFSET = GBRCUP_UB_TENSOR_OFFSET + 32 * UB_LINE_SIZE;
        constexpr uint32_t G_UB_HALF_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + 1 * UB_LINE_SIZE;

        aUbTensor = resource.ubBuf.template GetBufferByByte<half>(ATTN_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<AElementInput>(MASK_UB_TENSOR_OFFSET);
        srcUbTensor = resource.ubBuf.template GetBufferByByte<AElementOutput>(SRC_UB_TENSOR_OFFSET);
        floatUbTensor = resource.ubBuf.template GetBufferByByte<float>(FLOAT_UB_TENSOR_OFFSET);
        gbrcleftUbTensor = resource.ubBuf.template GetBufferByByte<AElementInput>(GBRCLEFT_UB_TENSOR_OFFSET);
        gbrcupUbTensor = resource.ubBuf.template GetBufferByByte<AElementInput>(GBRCUP_UB_TENSOR_OFFSET);
        gUbTensor = resource.ubBuf.template GetBufferByByte<GElementInput>(G_UB_TENSOR_OFFSET);
        gUbhalfTensor = resource.ubBuf.template GetBufferByByte<AElementInput>(G_UB_HALF_TENSOR_OFFSET);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<AElementOutput> maskOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<AElementInput> attnInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim
        )
    {
        uint32_t mActual = chunkSize;
        uint32_t nActual = chunkSize;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offset = mOffset * nActual + nOffset;

        AscendC::ResetMask();
        AscendC::GlobalTensor<AElementOutput> maskOutputThisSubBlock = maskOutput[offset];
        AscendC::GlobalTensor<AElementInput> attnInputThisSubBlock = attnInput[offset];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(mActualThisSubBlock * nActual * sizeof(half)), 0, 0, 0};
        AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, (mActualThisSubBlock * nActual + 16 - 1) / 16 * 16);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gUbTensor, gInputThisSubBlock, (mActual + 8 - 1) / 8 * 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(gUbhalfTensor, gUbTensor, AscendC::RoundMode::CAST_NONE, mActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int i = 0; i < mActualThisSubBlock; ++i) {
            for (int j = 0; j < nActual; ++j) {
                gbrcleftUbTensor.SetValue(i * nActual + j,
                    gUbhalfTensor.GetValue(i + ((subBlockIdx == 0) ? 0 : mActualPerSubBlock)));

                gbrcupUbTensor.SetValue(i * nActual + j, gUbhalfTensor.GetValue(j));
            }
        }
        
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sub(gbrcleftUbTensor, gbrcleftUbTensor, gbrcupUbTensor, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp(gbrcleftUbTensor, gbrcleftUbTensor, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Duplicate<half>(maskUbTensor, 0.0, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int i = 0; i < mActualThisSubBlock; ++i) {
            for (int j = 0; j < i + mOffset + 1; ++j) {
                maskUbTensor.SetValue(i * nActual + j, gbrcleftUbTensor.GetValue(i * nActual + j));
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(aUbTensor, aUbTensor, maskUbTensor, mActualThisSubBlock * nActual);
        if constexpr(!std::is_same<AElementOutput, half>::value) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(floatUbTensor, aUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(srcUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(maskOutputThisSubBlock, srcUbTensor, copyParams);
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(maskOutputThisSubBlock, aUbTensor, copyParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::LocalTensor<half> aUbTensor;
    AscendC::LocalTensor<half> maskUbTensor;
    AscendC::LocalTensor<half> gbrcleftUbTensor;
    AscendC::LocalTensor<half> gbrcupUbTensor;
    AscendC::LocalTensor<float> gUbTensor;
    AscendC::LocalTensor<half> gUbhalfTensor;
    AscendC::LocalTensor<AElementOutput> srcUbTensor;
    AscendC::LocalTensor<float> floatUbTensor;
};
}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP
/*
 * Copyright (C) 2022 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define LOG_TAG "SubGraphContext"

#include "SubGraphContext.h"

#include "FlatbufferModelBuilderUtils.h"

namespace android {
namespace nn {

SubGraphContext::SubGraphContext(const Model* model, const Model::Subgraph* subgraph,
                                 flatbuffers::FlatBufferBuilder* builder,
                                 std::vector<OpCodeFlatbuffer>* opCodesVector,
                                 std::vector<int>* opCodeIndexForOperationType,
                                 std::vector<BufferFlatbuffer>* bufferVector)
    : mModel(model),
      mSubgraph(subgraph),
      mBuilder(builder),
      mOpCodesVector(opCodesVector),
      mOpCodeIndexForOperationType(opCodeIndexForOperationType),
      mBufferVector(bufferVector) {
    CHECK(model != nullptr);
    CHECK(subgraph != nullptr);
    CHECK(opCodesVector != nullptr);
    CHECK(opCodeIndexForOperationType != nullptr);
    CHECK(bufferVector != nullptr);

    mOperandToTensorIdx.resize(subgraph->operands.size(), -1);
    mMappings.resize(model->pools.size());
}

SubGraphFlatbuffer SubGraphContext::finish() {
    return tflite::CreateSubGraphDirect(*mBuilder, &mTensorVector, &mInputTensors, &mOutputTensors,
                                        &mOperatorVector);
}

int SubGraphContext::addTensorFlatbuffer(TensorFlatbuffer tensor, int32_t operandIdx) {
    mTensorVector.push_back(tensor);

    int tensorIdx = mTensorVector.size() - 1;
    if (operandIdx >= 0) {
        CHECK(mOperandToTensorIdx[operandIdx] == -1);
        mOperandToTensorIdx[operandIdx] = tensorIdx;
    }
    return tensorIdx;
}

void SubGraphContext::addOperatorFlatbuffer(OperatorFlatbuffer opFlatbuffer) {
    mOperatorVector.push_back(opFlatbuffer);
}

void SubGraphContext::addSubGraphInput(int32_t operandIdx) {
    CHECK(mOperandToTensorIdx[operandIdx] != -1);
    mInputTensors.push_back(mOperandToTensorIdx[operandIdx]);
}

void SubGraphContext::addSubGraphOutput(int32_t operandIdx) {
    CHECK(mOperandToTensorIdx[operandIdx] != -1);
    mOutputTensors.push_back(mOperandToTensorIdx[operandIdx]);
}

uint32_t SubGraphContext::getOpCodeIndex(OperationType operationType) const {
    uint32_t idx = static_cast<uint32_t>(operationType);
    CHECK(mOpCodeIndexForOperationType->at(idx) != -1);
    return (*mOpCodeIndexForOperationType)[idx];
}

void SubGraphContext::addOpCode(OperationType operationType) {
    uint32_t idx = static_cast<uint32_t>(operationType);
    if (mOpCodeIndexForOperationType->at(idx) != -1) {
        return;
    }

    OperatorCodeFlatbuffer opCode;

    tflite::BuiltinOperator builtinCode = getFlatbufferOperator(operationType);
    if (builtinCode < tflite::BuiltinOperator::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES)
        opCode = tflite::CreateOperatorCode(
                *mBuilder, static_cast<int8_t>(builtinCode) /* deprecated_builtin_code */,
                0 /* custom_code */, getMaxOperatorVersionCode(builtinCode) /* version */);
    else
        opCode = tflite::CreateOperatorCode(*mBuilder, 0 /* deprecated_builtin_code */,
                                            0 /* custom_code */,
                                            getMaxOperatorVersionCode(builtinCode) /* version */,
                                            builtinCode /* builtin_code */);

    mOpCodesVector->push_back(opCode);
    (*mOpCodeIndexForOperationType)[idx] = mOpCodesVector->size() - 1;
}

int SubGraphContext::getTensorIdxFromOperandIdx(int operandIdx) const {
    return mOperandToTensorIdx[operandIdx];
}

const Mapping& SubGraphContext::getMapping(uint32_t poolIndex) {
    if (mMappings[poolIndex].size > 0) {
        return mMappings[poolIndex];
    }

    SharedMemory memory = mModel->pools[poolIndex];
    GeneralResult<Mapping> mapping = map(memory);
    CHECK(mapping.has_value()) << "CONSTANT_REFERENCE memory mapping error: "
                               << mapping.error().message;

    mMappings[poolIndex] = std::move(mapping).value();
    return mMappings[poolIndex];
}

std::pair<const uint8_t*, uint32_t> SubGraphContext::getConstantPointerAndLength(
        const Operand& operand) {
    CHECK(isOperandConstant(operand));

    if (operand.lifetime == Operand::LifeTime::CONSTANT_COPY) {
        return std::make_pair(mModel->operandValues.data() + operand.location.offset,
                              operand.location.length);
    }

    const Mapping& mapping = getMapping(operand.location.poolIndex);
    const uint8_t* memoryPtr = static_cast<const uint8_t*>(
            std::visit([](auto ptr) { return static_cast<const void*>(ptr); }, mapping.pointer));

    return std::make_pair(memoryPtr + operand.location.offset, operand.location.length);
}

uint32_t SubGraphContext::addBufferFromData(const uint8_t* data, uint32_t length) {
    auto dataVectorFlatbuffer = mBuilder->CreateVector(data, length);

    auto buffer = tflite::CreateBuffer(*mBuilder, dataVectorFlatbuffer);
    mBufferVector->push_back(buffer);

    return mBufferVector->size() - 1;
}

void SubGraphContext::createTensorFlatbufferFromOperand(uint32_t operandIdx) {
    // An output Operand to one Operation can be an input Operand to
    // another Operation, so this function can be run more than once.
    // We simply return if the Tensor for the Operand is already created.
    if (mOperandToTensorIdx[operandIdx] != -1) return;

    const Operand& operand = mSubgraph->operands[operandIdx];

    std::vector<float> scaleVector{operand.scale};
    std::vector<int64_t> zeroPointVector{operand.zeroPoint};

    // build quantization parameters
    auto quantizationParams = tflite::CreateQuantizationParametersDirect(
            *mBuilder, 0 /* min */, 0 /* max */, &scaleVector /* scale */,
            &zeroPointVector /* zero_point */,
            tflite::QuantizationDetails::QuantizationDetails_NONE /* details_type */);

    // add buffer if constant operand
    // buffer at index 0 is reserved for tensors without a buffer
    uint32_t bufferIdx = 0;
    if (isOperandConstant(operand)) {
        auto [data, dataLength] = getConstantPointerAndLength(operand);
        bufferIdx = addBufferFromData(data, dataLength);
    }

    // shape of tensor
    std::vector<int32_t> shape(operand.dimensions.begin(), operand.dimensions.end());
    replaceZeroDimensions(&shape);

    // build tensor
    TensorFlatbuffer tensor = tflite::CreateTensorDirect(
            *mBuilder, &shape, getTensorFlatbufferOperandType(operand.type) /* type */,
            bufferIdx /* buffer */, 0 /* name */, quantizationParams /* quantization */);
    addTensorFlatbuffer(tensor, operandIdx);
}

}  // namespace nn
}  // namespace android
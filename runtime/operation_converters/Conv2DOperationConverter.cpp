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

#include "Conv2DOperationConverter.h"

#include <vector>

#include "OperationConverterResolver.h"
#include "SubGraphContext.h"

namespace android {
namespace nn {

std::vector<int32_t> Conv2DOperationConverter::getConv2DInputs(const Operation& operation,
                                                               SubGraphContext* context) const {
    context->createTensorFlatbufferFromOperand(operation.inputs[kInputTensorIdx]);
    context->createTensorFlatbufferFromOperand(operation.inputs[kFilterTensorIdx]);
    context->createTensorFlatbufferFromOperand(operation.inputs[kBiasTensorIdx]);
    return {context->getTensorIdxFromOperandIdx(operation.inputs[kInputTensorIdx]),
            context->getTensorIdxFromOperandIdx(operation.inputs[kFilterTensorIdx]),
            context->getTensorIdxFromOperandIdx(operation.inputs[kBiasTensorIdx])};
}

std::vector<int32_t> Conv2DOperationConverter::getConv2DOutputs(const Operation& operation,
                                                                SubGraphContext* context) const {
    context->createTensorFlatbufferFromOperand(operation.outputs[kOutputTensorIdx]);
    return {context->getTensorIdxFromOperandIdx(operation.outputs[kOutputTensorIdx])};
}

int Conv2DOperationConverter::decomposeExplicitPadding(const Operation& operation,
                                                       SubGraphContext* context) const {
    const Model::Subgraph* subgraph = context->getSubgraph();
    const Operand& inputOperand = subgraph->operands[operation.inputs[0]];

    const Dimensions& dims = inputOperand.dimensions;
    if (dims[1] == 0 || dims[2] == 0) {
        LOG(WARNING) << "Unknown height and width dimensions for input not supported";
        return -1;
    }

    // pad options
    auto padOptionsFlatbuffer = tflite::CreatePadOptions(context->getBuilder());

    // check to make sure padding Operands are constants
    const Operand& frontWidthPaddingOperand = subgraph->operands[operation.inputs[3]];
    const Operand& backWidthPaddingOperand = subgraph->operands[operation.inputs[4]];
    const Operand& frontHeightPaddingOperand = subgraph->operands[operation.inputs[5]];
    const Operand& backHeightPaddingOperand = subgraph->operands[operation.inputs[6]];
    if (!isOperandConstant(frontWidthPaddingOperand) ||
        !isOperandConstant(backWidthPaddingOperand) ||
        !isOperandConstant(frontHeightPaddingOperand) ||
        !isOperandConstant(backHeightPaddingOperand)) {
        LOG(WARNING) << "At least one Padding Operand is not a constant";
        return -1;
    }

    // get padding params
    int32_t frontHeightPadding = context->getConstantScalar<int32_t>(frontHeightPaddingOperand);
    int32_t backHeightPadding = context->getConstantScalar<int32_t>(backHeightPaddingOperand);
    int32_t frontWidthPadding = context->getConstantScalar<int32_t>(frontWidthPaddingOperand);
    int32_t backWidthPadding = context->getConstantScalar<int32_t>(backWidthPaddingOperand);

    // build padding buffer
    int numDimensionsInput = static_cast<int>(dims.size());
    std::vector<int32_t> paddingData(numDimensionsInput * 2, 0);
    paddingData[2] = frontHeightPadding;
    paddingData[3] = backHeightPadding;
    paddingData[4] = frontWidthPadding;
    paddingData[5] = backWidthPadding;
    uint32_t paddingBufferIdx = context->addBufferFromData(
            reinterpret_cast<uint8_t*>(paddingData.data()), paddingData.size() * sizeof(int32_t));

    // create new tensor for padding
    std::vector<int32_t> padShape{numDimensionsInput, 2};
    auto padTensor = tflite::CreateTensorDirect(context->getBuilder(), &padShape /* shape */,
                                                tflite::TensorType::TensorType_INT32 /* type */,
                                                paddingBufferIdx /* buffer */);
    int padTensorIdx = context->addTensorFlatbuffer(padTensor);

    // add inputs for padding operation
    std::vector<int32_t> padInputs = {context->getTensorIdxFromOperandIdx(operation.inputs[0]),
                                      padTensorIdx};

    // add opcode for pad if it does not exist yet
    context->addOpCode(OperationType::PAD);

    // get dimensions of output of pad operation
    std::vector<int32_t> padToConv2dShape(dims.begin(), dims.end());
    padToConv2dShape[1] = frontHeightPadding + padToConv2dShape[1] + backHeightPadding;
    padToConv2dShape[2] = frontWidthPadding + padToConv2dShape[2] + backWidthPadding;
    replaceZeroDimensions(&padToConv2dShape);

    // create new tensor to be output of pad & input for conv2d
    auto padToConv2dTensor = tflite::CreateTensorDirect(
            context->getBuilder(), &padToConv2dShape /* shape */,
            getTensorFlatbufferOperandType(subgraph->operands[operation.inputs[0]].type) /* type */,
            0 /* buffer */);
    int padToConv2dTensorIdx = context->addTensorFlatbuffer(padToConv2dTensor);

    // set output for padding operation and add to operators
    std::vector<int32_t> padOutputs{padToConv2dTensorIdx};

    OperatorFlatbuffer padOp = tflite::CreateOperatorDirect(
            context->getBuilder(), context->getOpCodeIndex(OperationType::PAD), &padInputs,
            &padOutputs, tflite::BuiltinOptions::BuiltinOptions_PadOptions,
            padOptionsFlatbuffer.Union());
    context->addOperatorFlatbuffer(padOp);

    // Return tensor index of pad output created
    return padToConv2dTensorIdx;
}

bool Conv2DOperationConverter::convert(const Operation& operation, SubGraphContext* context) const {
    const Model::Subgraph* subgraph = context->getSubgraph();

    // if there are less than 8 inputs or the input at the 7th index is a BOOL, there is implicit
    // padding
    bool isImplicitPadding = false;
    if (operation.inputs.size() < 8 ||
        subgraph->operands[operation.inputs[7]].type == OperandType::BOOL) {
        isImplicitPadding = true;
    }

    std::vector<int32_t> inputs = getConv2DInputs(operation, context);
    std::vector<int32_t> outputs = getConv2DOutputs(operation, context);

    // if explicit padding, we need to decompose the operation to a separate padding op and a conv2d
    // op
    if (!isImplicitPadding) {
        int padOpIdx = decomposeExplicitPadding(operation, context);
        if (padOpIdx == -1) {
            LOG(WARNING) << "decomposeExplicitPadding unsuccessful";
            return false;
        }
        inputs[0] = padOpIdx;
    }

    int baseOptionsIdx = 4;
    tflite::Padding padding;
    if (isImplicitPadding) {
        const Operand& paddingTypeOperand = subgraph->operands[operation.inputs[3]];
        if (!isOperandConstant(paddingTypeOperand)) {
            LOG(WARNING) << "paddingTypeOperand is not a constant: " << operation.inputs[3];
            return false;
        }

        int32_t paddingType = context->getConstantScalar<int32_t>(paddingTypeOperand);
        padding = getTFLitePadding(paddingType);
    } else {
        padding = tflite::Padding::Padding_VALID;
        baseOptionsIdx = 7;
    }

    // check if stride and activation Operands are constant
    const Operand& strideWOperand =
            subgraph->operands[operation.inputs[baseOptionsIdx + kStrideWOffset]];
    const Operand& strideHOperand =
            subgraph->operands[operation.inputs[baseOptionsIdx + kStrideHOffset]];
    const Operand& activationOperand =
            subgraph->operands[operation.inputs[baseOptionsIdx + kActivationOffset]];
    if (!isOperandConstant(strideWOperand) || !isOperandConstant(strideHOperand) ||
        !isOperandConstant(activationOperand)) {
        LOG(WARNING) << "strideWOperand, strideHOperand, or activationOperand is not a constant";
        return false;
    }

    // get strides and activation
    int32_t strideW = context->getConstantScalar<int32_t>(strideWOperand);
    int32_t strideH = context->getConstantScalar<int32_t>(strideHOperand);
    int32_t activation = context->getConstantScalar<int32_t>(activationOperand);

    // check for nchw
    bool isNchw = false;
    int isNchwIdx = baseOptionsIdx + kIsNchwOffset;
    if (operation.inputs.size() > static_cast<uint32_t>(isNchwIdx)) {
        const Operand& isNchwOperand = subgraph->operands[operation.inputs[isNchwIdx]];
        if (!isOperandConstant(isNchwOperand)) {
            LOG(WARNING) << "isNchwOperand is not a constant";
            return false;
        }

        isNchw = context->getConstantScalar<bool>(isNchwOperand);
        if (isNchw) {
            LOG(WARNING) << "TFLite does not support NCHW formatted input tensors";
            return false;
        }
    }

    // dilations
    int dilationWIdx = baseOptionsIdx + kDilationWOffset;
    int dilationHIdx = baseOptionsIdx + kDilationHOffset;
    // default dilation factors are 1
    int32_t dilationW = 1;
    int32_t dilationH = 1;
    if (operation.inputs.size() > static_cast<uint32_t>(dilationWIdx)) {
        const Operand& dilationWOperand = subgraph->operands[operation.inputs[dilationWIdx]];
        if (!isOperandConstant(dilationWOperand)) {
            LOG(WARNING) << "dilationWOperand is not a constant: "
                         << operation.inputs[dilationWIdx];
            return false;
        }

        dilationW = context->getConstantScalar<int32_t>(dilationWOperand);
    }
    if (operation.inputs.size() > static_cast<uint32_t>(dilationHIdx)) {
        const Operand& dilationHOperand = subgraph->operands[operation.inputs[dilationHIdx]];
        if (!isOperandConstant(dilationHOperand)) {
            LOG(WARNING) << "dilationHOperand is not a constant: "
                         << operation.inputs[dilationHIdx];
            return false;
        }

        dilationH = context->getConstantScalar<int32_t>(dilationHOperand);
    }

    flatbuffers::Offset<tflite::Conv2DOptions> optionsFlatbuffer = tflite::CreateConv2DOptions(
            context->getBuilder(), padding, strideW, strideH,
            static_cast<tflite::ActivationFunctionType>(activation) /* fused_activation_function */,
            dilationW, dilationH);
    auto operatorFlatbuffer = tflite::CreateOperatorDirect(
            context->getBuilder() /* builder */,
            context->getOpCodeIndex(operation.type) /* opcode_index */, &inputs /* inputs */,
            &outputs /* outputs */,
            tflite::BuiltinOptions::BuiltinOptions_Conv2DOptions /* builtin_options_type */,
            optionsFlatbuffer.Union() /* builtin_options */);
    context->addOperatorFlatbuffer(operatorFlatbuffer);

    return true;
}

NN_REGISTER_OPERATION_CONVERTER(CONV_2D, Conv2DOperationConverter);

}  // namespace nn
}  // namespace android
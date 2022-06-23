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

#ifndef ANDROID_PACKAGES_MODULES_NEURALNETWORKS_RUNTIME_FLATBUFFER_MODEL_BUILDER_UTILS_H
#define ANDROID_PACKAGES_MODULES_NEURALNETWORKS_RUNTIME_FLATBUFFER_MODEL_BUILDER_UTILS_H

#include <tensorflow/lite/schema/schema_generated.h>

#include "NeuralNetworks.h"
#include "TypeManager.h"

namespace android {
namespace nn {

using OpCodeFlatbuffer = flatbuffers::Offset<tflite::OperatorCode>;
using OpCodesFlatbuffer = flatbuffers::Offset<flatbuffers::Vector<OpCodeFlatbuffer>>;

using SubGraphFlatbuffer = flatbuffers::Offset<tflite::SubGraph>;
using SubGraphsFlatbuffer = flatbuffers::Offset<flatbuffers::Vector<SubGraphFlatbuffer>>;

using OperatorFlatbuffer = flatbuffers::Offset<tflite::Operator>;
using OperatorsFlatbuffer = flatbuffers::Offset<flatbuffers::Vector<OperatorFlatbuffer>>;

using TensorFlatbuffer = flatbuffers::Offset<tflite::Tensor>;
using TensorsFlatbuffer = flatbuffers::Offset<flatbuffers::Vector<TensorFlatbuffer>>;

using BufferFlatbuffer = flatbuffers::Offset<tflite::Buffer>;

using ModelFlatbuffer = flatbuffers::Offset<tflite::Model>;

// Only supports tensor types
// Will crash if passed in a scalar type
inline tflite::TensorType getTensorFlatbufferOperandType(const OperandType& type) {
    CHECK(TypeManager::get()->isTensorType(type));

    // TODO: Map more operands
    switch (type) {
        case OperandType::TENSOR_FLOAT32:
            return tflite::TensorType::TensorType_FLOAT32;
        case OperandType::TENSOR_FLOAT16:
            return tflite::TensorType::TensorType_FLOAT16;
        case OperandType::TENSOR_INT32:
            return tflite::TensorType::TensorType_INT32;
        case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
            return tflite::TensorType::TensorType_INT8;
        default:
            LOG(FATAL) << "OperandType not supported: " << type;
            return {};
    }
}

inline tflite::BuiltinOperator getFlatbufferOperator(const OperationType& type) {
    // TODO: Add more operation types
    switch (type) {
        case OperationType::PAD:
            return tflite::BuiltinOperator::BuiltinOperator_PAD;
        case OperationType::CONV_2D:
            return tflite::BuiltinOperator::BuiltinOperator_CONV_2D;
        default:
            LOG(FATAL) << "OperationType not supported: " << type;
            return {};
    }
}

}  // namespace nn
}  // namespace android

#endif  // ANDROID_PACKAGES_MODULES_NEURALNETWORKS_RUNTIME_FLATBUFFER_MODEL_BUILDER_UTILS_H

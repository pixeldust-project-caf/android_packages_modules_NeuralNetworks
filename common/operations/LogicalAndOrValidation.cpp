/*
 * Copyright (C) 2018 The Android Open Source Project
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

#include "LogicalAndOr.h"
#include "OperationsUtils.h"

namespace android::nn {
namespace logical {

Result<Version> validate(const IOperationValidationContext* context) {
    NN_RET_CHECK_EQ(context->getNumInputs(), kNumInputs);
    NN_RET_CHECK_EQ(context->getNumOutputs(), kNumOutputs);
    OperandType inputType = context->getInputType(kInputTensor1);
    NN_RET_CHECK(inputType == OperandType::TENSOR_BOOL8)
            << "Unsupported tensor type for a logical operation";
    NN_RET_CHECK(validateInputTypes(context, {inputType, inputType}));
    NN_RET_CHECK(validateOutputTypes(context, {inputType}));
    return kVersionFeatureLevel3;
}

}  // namespace logical

NN_DEFINE_VALIDATION_FUNCTION(LOGICAL_AND, logical::validate);
NN_DEFINE_VALIDATION_FUNCTION(LOGICAL_OR, logical::validate);

}  // namespace android::nn

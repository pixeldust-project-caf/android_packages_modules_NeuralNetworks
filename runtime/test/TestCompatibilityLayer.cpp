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

#include <android-base/logging.h>
#include <android-base/properties.h>
#include <ftw.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "AndroidVersionUtil.h"
#include "FlatbufferModelBuilder.h"
#include "GeneratedTestUtils.h"
#include "ModelBuilder.h"
#include "NeuralNetworks.h"
#include "NeuralNetworksTypes.h"
#include "TestHarness.h"
#include "TestNeuralNetworksWrapper.h"
#include "TestUtils.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#pragma clang diagnostic pop

#ifdef NNTEST_CTS
#define NNTEST_COMPUTE_MODE
#endif

namespace android::nn::generated_tests {
using namespace test_wrapper;
using namespace test_helper;

class CompatibilityLayerGeneratedTests : public GeneratedTestBase {
   protected:
    void SetUp() override;
    void TearDown() override;

    // Test driver for those generated from packages/modules/NeuralNetworks/runtime/test/specs
    void execute(const TestModel& testModel);

    bool mTestDynamicOutputShape = false;
    bool mTestSupported = true;
};

class CompatibilityLayerGeneratedTestsSupported : public CompatibilityLayerGeneratedTests {};
class CompatibilityLayerGeneratedTestsUnsupported : public CompatibilityLayerGeneratedTests {};
class CompatibilityLayerGeneratedTestsDynamicOutput : public CompatibilityLayerGeneratedTests {};

void CompatibilityLayerGeneratedTests::execute(const TestModel& testModel) {
    GeneratedModel model;
    createModel(testModel, mTestDynamicOutputShape, &model);
    if (testModel.expectFailure && !model.isValid()) {
        return;
    }
    ASSERT_EQ(model.finish(), test_wrapper::Result::NO_ERROR);
    ASSERT_TRUE(model.isValid());

    auto flatbufferModelBuilder = reinterpret_cast<FlatbufferModelBuilder*>(model.getHandle());
    // Load the model
    auto tfliteModel = flatbufferModelBuilder->createTfliteModel();
    if (!mTestSupported) {
        ASSERT_FALSE(tfliteModel.ok());
        return;
    }
    ASSERT_TRUE(tfliteModel.ok()) << tfliteModel.error();

    std::unique_ptr<tflite::FlatBufferModel> flatBufferModel =
            tflite::FlatBufferModel::BuildFromModel(tfliteModel.value());
    ASSERT_NE(flatBufferModel, nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    ASSERT_EQ(tflite::InterpreterBuilder(*flatBufferModel, resolver)(&interpreter), kTfLiteOk);
    ASSERT_NE(interpreter, nullptr);

    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    for (uint32_t i = 0; i < testModel.main.inputIndexes.size(); i++) {
        const auto& operand = testModel.main.operands[testModel.main.inputIndexes[i]];

        ASSERT_LE(interpreter->input_tensor(i)->bytes, operand.data.size());
        std::memcpy(interpreter->input_tensor(i)->data.raw, operand.data.get<void>(),
                    operand.data.size());
    }

    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    std::vector<TestBuffer> outputs;
    for (uint32_t i = 0; i < testModel.main.outputIndexes.size(); i++) {
        const auto& operand = testModel.main.operands[testModel.main.outputIndexes[i]];
        const size_t bufferSize = std::max<size_t>(operand.data.size(), 1);
        ASSERT_GE(interpreter->output_tensor(i)->bytes, bufferSize);
        outputs.emplace_back(bufferSize, interpreter->output_tensor(i)->data.raw);
    }

    checkResults(testModel, outputs);
}

void CompatibilityLayerGeneratedTests::SetUp() {
    GeneratedTestBase::SetUp();
}

void CompatibilityLayerGeneratedTests::TearDown() {
    GeneratedTestBase::TearDown();
}

namespace {

bool compatibleTest(const TestModel& testModel) {
    static const std::vector<TestOperationType> kSupportedOperationTypes{
            TestOperationType::CONV_2D};
    static const std::vector<TestOperandType> kSupportedOperandTypes{
            TestOperandType::TENSOR_FLOAT32, TestOperandType::TENSOR_INT32,
            TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED, TestOperandType::BOOL,
            TestOperandType::INT32};

    if (testModel.hasControlFlow()) {
        return false;
    }

    bool result = true;
    const TestSubgraph& mainSubgraph = testModel.main;

    result &= std::all_of(
            mainSubgraph.operations.begin(), mainSubgraph.operations.end(),
            [&mainSubgraph](const TestOperation& operation) {
                bool isOperationCompatible = true;
                // ensure that tensors are nhwc and filter is constant
                if (operation.type == TestOperationType::CONV_2D) {
                    int isNchwIdx = 10;
                    if (operation.inputs.size() < 8 ||
                        mainSubgraph.operands[operation.inputs[7]].type == TestOperandType::BOOL) {
                        isNchwIdx = 7;
                    }

                    if (operation.inputs.size() > static_cast<uint32_t>(isNchwIdx)) {
                        isOperationCompatible &=
                                !(*mainSubgraph.operands[operation.inputs[isNchwIdx]]
                                           .data.get<bool>());
                    }

                    const int kFilterIdx = 1;
                    TestOperandLifeTime filterLifetime =
                            mainSubgraph.operands[operation.inputs[kFilterIdx]].lifetime;
                    isOperationCompatible &=
                            (filterLifetime == TestOperandLifeTime::CONSTANT_COPY) ||
                            (filterLifetime == TestOperandLifeTime::CONSTANT_REFERENCE);
                }

                isOperationCompatible &=
                        std::find(kSupportedOperationTypes.begin(), kSupportedOperationTypes.end(),
                                  operation.type) != kSupportedOperationTypes.end();

                return isOperationCompatible;
            });

    result &= std::all_of(mainSubgraph.operands.begin(), mainSubgraph.operands.end(),
                          [](const TestOperand& operand) {
                              return std::find(kSupportedOperandTypes.begin(),
                                               kSupportedOperandTypes.end(),
                                               operand.type) != kSupportedOperandTypes.end();
                          });

    return result;
}

}  // namespace

TEST_P(CompatibilityLayerGeneratedTestsSupported, CompatibilityLayerSupported) {
    mTestSupported = true;
    execute(testModel);
}

TEST_P(CompatibilityLayerGeneratedTestsUnsupported, CompatibilityLayerUnsupported) {
    mTestSupported = false;
    execute(testModel);
}

TEST_P(CompatibilityLayerGeneratedTestsDynamicOutput, CompatibilityLayerDynamicOutput) {
    mTestDynamicOutputShape = true;
    mTestSupported = false;
    execute(testModel);
}

INSTANTIATE_GENERATED_TEST(CompatibilityLayerGeneratedTestsSupported,
                           [](const TestModel& testModel) {
                               return !testModel.expectFailure && compatibleTest(testModel);
                           });

INSTANTIATE_GENERATED_TEST(CompatibilityLayerGeneratedTestsUnsupported,
                           [](const TestModel& testModel) {
                               return !testModel.expectFailure && !compatibleTest(testModel);
                           });

INSTANTIATE_GENERATED_TEST(CompatibilityLayerGeneratedTestsDynamicOutput,
                           [](const TestModel& testModel) {
                               return !testModel.expectFailure && !testModel.hasScalarOutputs();
                           });

}  // namespace android::nn::generated_tests
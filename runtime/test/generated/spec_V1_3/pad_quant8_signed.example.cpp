// Generated from pad_quant8_signed.mod.py
// DO NOT EDIT
// clang-format off
#include "TestHarness.h"
using namespace test_helper;

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_quant8_signed() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128, -127, -127, -126, -126, -125}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({1, 2, 3, 4, 3, 3, 2, 1}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -128, -128, -128, -126, -126, -125, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128}),
                .dimensions = {4, 8, 8, 6},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }},
        .operations = {{
                .inputs = {0, 1},
                .outputs = {2},
                .type = TestOperationType::PAD
            }},
        .outputIndexes = {2}
    };
    return model;
}

const auto dummy_test_model_quant8_signed = TestModelManager::get().add("pad_quant8_signed_quant8_signed", get_test_model_quant8_signed());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_quant8_signed_all_inputs_as_internal() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {3},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({1, 2, 3, 4, 3, 3, 2, 1}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -128, -128, -128, -126, -126, -125, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128}),
                .dimensions = {4, 8, 8, 6},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128, -127, -127, -126, -126, -125}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {3, 4, 5},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1},
                .outputs = {2},
                .type = TestOperationType::PAD
            }},
        .outputIndexes = {2}
    };
    return model;
}

const auto dummy_test_model_quant8_signed_all_inputs_as_internal = TestModelManager::get().add("pad_quant8_signed_quant8_signed_all_inputs_as_internal", get_test_model_quant8_signed_all_inputs_as_internal());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({3, 1}),
                .dimensions = {1, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128, -128, -128, -127, -126, -125, -128}),
                .dimensions = {7},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }},
        .operations = {{
                .inputs = {0, 1},
                .outputs = {2},
                .type = TestOperationType::PAD
            }},
        .outputIndexes = {2}
    };
    return model;
}

const auto dummy_test_model = TestModelManager::get().add("pad_quant8_signed", get_test_model());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_inputs_as_internal() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {3},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({3, 1}),
                .dimensions = {1, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128, -128, -128, -127, -126, -125, -128}),
                .dimensions = {7},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {3, 4, 5},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1},
                .outputs = {2},
                .type = TestOperationType::PAD
            }},
        .outputIndexes = {2}
    };
    return model;
}

const auto dummy_test_model_all_inputs_as_internal = TestModelManager::get().add("pad_quant8_signed_all_inputs_as_internal", get_test_model_all_inputs_as_internal());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_2() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0, 0, 0, 2, 1, 3, 0, 0}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128, -127, -126, -125, -128, -128, -128, -128, -124, -123, -122, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128}),
                .dimensions = {1, 4, 7, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }},
        .operations = {{
                .inputs = {0, 1},
                .outputs = {2},
                .type = TestOperationType::PAD
            }},
        .outputIndexes = {2}
    };
    return model;
}

const auto dummy_test_model_2 = TestModelManager::get().add("pad_quant8_signed_2", get_test_model_2());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_inputs_as_internal_2() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {3},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0, 0, 0, 2, 1, 3, 0, 0}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128, -127, -126, -125, -128, -128, -128, -128, -124, -123, -122, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128}),
                .dimensions = {1, 4, 7, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-128}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -128
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {3, 4, 5},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1},
                .outputs = {2},
                .type = TestOperationType::PAD
            }},
        .outputIndexes = {2}
    };
    return model;
}

const auto dummy_test_model_all_inputs_as_internal_2 = TestModelManager::get().add("pad_quant8_signed_all_inputs_as_internal_2", get_test_model_all_inputs_as_internal_2());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_3() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -119
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0, 0, 0, 2, 1, 3, 0, 0}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-119, -127, -126, -125, -119, -119, -119, -119, -124, -123, -122, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119}),
                .dimensions = {1, 4, 7, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -119
            }},
        .operations = {{
                .inputs = {0, 1},
                .outputs = {2},
                .type = TestOperationType::PAD
            }},
        .outputIndexes = {2}
    };
    return model;
}

const auto dummy_test_model_3 = TestModelManager::get().add("pad_quant8_signed_3", get_test_model_3());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_inputs_as_internal_3() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {3},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -119
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0, 0, 0, 2, 1, 3, 0, 0}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-119, -127, -126, -125, -119, -119, -119, -119, -124, -123, -122, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119}),
                .dimensions = {1, 4, 7, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -119
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -119
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-119}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -119
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {3, 4, 5},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1},
                .outputs = {2},
                .type = TestOperationType::PAD
            }},
        .outputIndexes = {2}
    };
    return model;
}

const auto dummy_test_model_all_inputs_as_internal_3 = TestModelManager::get().add("pad_quant8_signed_all_inputs_as_internal_3", get_test_model_all_inputs_as_internal_3());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_4() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0, 0, 0, 2, 1, 3, 0, 0}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({9}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({9, -127, -126, -125, 9, 9, 9, 9, -124, -123, -122, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}),
                .dimensions = {1, 4, 7, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }},
        .operations = {{
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_4 = TestModelManager::get().add("pad_quant8_signed_4", get_test_model_4());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_inputs_as_internal_4() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {4},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0, 0, 0, 2, 1, 3, 0, 0}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({9}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({9, -127, -126, -125, 9, 9, 9, 9, -124, -123, -122, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}),
                .dimensions = {1, 4, 7, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-124}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {4, 5, 6},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_inputs_as_internal_4 = TestModelManager::get().add("pad_quant8_signed_all_inputs_as_internal_4", get_test_model_all_inputs_as_internal_4());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_tensors_as_inputs() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0, 1},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0, 0, 0, 2, 1, 3, 0, 0}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({9}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({9, -127, -126, -125, 9, 9, 9, 9, -124, -123, -122, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}),
                .dimensions = {1, 4, 7, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }},
        .operations = {{
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_tensors_as_inputs = TestModelManager::get().add("pad_quant8_signed_all_tensors_as_inputs", get_test_model_all_tensors_as_inputs());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_tensors_as_inputs_all_inputs_as_internal() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {1, 4},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0, 0, 0, 2, 1, 3, 0, 0}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({9}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({9, -127, -126, -125, 9, 9, 9, 9, -124, -123, -122, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}),
                .dimensions = {1, 4, 7, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 2, 3, 1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-124}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {4, 5, 6},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_tensors_as_inputs_all_inputs_as_internal = TestModelManager::get().add("pad_quant8_signed_all_tensors_as_inputs_all_inputs_as_internal", get_test_model_all_tensors_as_inputs_all_inputs_as_internal());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_5() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({1, 2, 3, 4, 3, 3, 2, 1}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({-125}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -127, -126, -125, -125, -125, -125, -124, -123, -122, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125}),
                .dimensions = {4, 8, 8, 6},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }},
        .operations = {{
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_5 = TestModelManager::get().add("pad_quant8_signed_5", get_test_model_5());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_inputs_as_internal_5() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {4},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({1, 2, 3, 4, 3, 3, 2, 1}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({-125}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -127, -126, -125, -125, -125, -125, -124, -123, -122, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125}),
                .dimensions = {4, 8, 8, 6},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-124}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {4, 5, 6},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_inputs_as_internal_5 = TestModelManager::get().add("pad_quant8_signed_all_inputs_as_internal_5", get_test_model_all_inputs_as_internal_5());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_tensors_as_inputs_2() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0, 1},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({1, 2, 3, 4, 3, 3, 2, 1}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({-125}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -127, -126, -125, -125, -125, -125, -124, -123, -122, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125}),
                .dimensions = {4, 8, 8, 6},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }},
        .operations = {{
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_tensors_as_inputs_2 = TestModelManager::get().add("pad_quant8_signed_all_tensors_as_inputs_2", get_test_model_all_tensors_as_inputs_2());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_tensors_as_inputs_all_inputs_as_internal_2() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {1, 4},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({1, 2, 3, 4, 3, 3, 2, 1}),
                .dimensions = {4, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({-125}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -127, -126, -125, -125, -125, -125, -124, -123, -122, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125}),
                .dimensions = {4, 8, 8, 6},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125, -124, -123, -122}),
                .dimensions = {1, 1, 2, 3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-124}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {4, 5, 6},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_tensors_as_inputs_all_inputs_as_internal_2 = TestModelManager::get().add("pad_quant8_signed_all_tensors_as_inputs_all_inputs_as_internal_2", get_test_model_all_tensors_as_inputs_all_inputs_as_internal_2());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_6() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({3, 1}),
                .dimensions = {1, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({9}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({9, 9, 9, -127, -126, -125, 9}),
                .dimensions = {7},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }},
        .operations = {{
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_6 = TestModelManager::get().add("pad_quant8_signed_6", get_test_model_6());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_inputs_as_internal_6() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {4},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({3, 1}),
                .dimensions = {1, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({9}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({9, 9, 9, -127, -126, -125, 9}),
                .dimensions = {7},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-124}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {4, 5, 6},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_inputs_as_internal_6 = TestModelManager::get().add("pad_quant8_signed_all_inputs_as_internal_6", get_test_model_all_inputs_as_internal_6());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_tensors_as_inputs_3() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {0, 1},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({3, 1}),
                .dimensions = {1, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({9}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({9, 9, 9, -127, -126, -125, 9}),
                .dimensions = {7},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }},
        .operations = {{
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_tensors_as_inputs_3 = TestModelManager::get().add("pad_quant8_signed_all_tensors_as_inputs_3", get_test_model_all_tensors_as_inputs_3());

}  // namespace generated_tests::pad_quant8_signed

namespace generated_tests::pad_quant8_signed {

const TestModel& get_test_model_all_tensors_as_inputs_all_inputs_as_internal_3() {
    static TestModel model = {
        .expectFailure = false,
        .expectedMultinomialDistributionTolerance = 0,
        .inputIndexes = {1, 4},
        .isRelaxed = false,
        .minSupportedVersion = TestHalVersion::V1_3,
        .operands = {{
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::TEMPORARY_VARIABLE,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({3, 1}),
                .dimensions = {1, 2},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::TENSOR_INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({9}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({9, 9, 9, -127, -126, -125, 9}),
                .dimensions = {7},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_OUTPUT,
                .numberOfConsumers = 0,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-127, -126, -125}),
                .dimensions = {3},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::SUBGRAPH_INPUT,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int8_t>({-124}),
                .dimensions = {1},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 2.3f,
                .type = TestOperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                .zeroPoint = -124
            }, {
                .channelQuant = {},
                .data = TestBuffer::createFromVector<int32_t>({0}),
                .dimensions = {},
                .isIgnored = false,
                .lifetime = TestOperandLifeTime::CONSTANT_COPY,
                .numberOfConsumers = 1,
                .scale = 0.0f,
                .type = TestOperandType::INT32,
                .zeroPoint = 0
            }},
        .operations = {{
                .inputs = {4, 5, 6},
                .outputs = {0},
                .type = TestOperationType::ADD
            }, {
                .inputs = {0, 1, 2},
                .outputs = {3},
                .type = TestOperationType::PAD_V2
            }},
        .outputIndexes = {3}
    };
    return model;
}

const auto dummy_test_model_all_tensors_as_inputs_all_inputs_as_internal_3 = TestModelManager::get().add("pad_quant8_signed_all_tensors_as_inputs_all_inputs_as_internal_3", get_test_model_all_tensors_as_inputs_all_inputs_as_internal_3());

}  // namespace generated_tests::pad_quant8_signed

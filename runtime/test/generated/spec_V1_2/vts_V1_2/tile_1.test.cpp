// Generated from tile_1.mod.py
// DO NOT EDIT
// clang-format off
#include "GeneratedTests.h"

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel();
bool is_ignored(int);

TEST_F(NeuralnetworksHidlTest, tile_1) {
  Execute(device,
          createTestModel,
          is_ignored,
          ::generated_tests::tile_1::get_examples());
}

TEST_F(ValidationTest, tile_1) {
  const Model model = createTestModel();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_dynamic_output_shape();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_dynamic_output_shape();
bool is_ignored_dynamic_output_shape(int);

TEST_F(DynamicOutputShapeTest, tile_1_dynamic_output_shape) {
  Execute(device,
          createTestModel_dynamic_output_shape,
          is_ignored_dynamic_output_shape,
          ::generated_tests::tile_1::get_examples_dynamic_output_shape(), true);
}

TEST_F(ValidationTest, tile_1_dynamic_output_shape) {
  const Model model = createTestModel_dynamic_output_shape();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_dynamic_output_shape());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_all_inputs_as_internal();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_all_inputs_as_internal();
bool is_ignored_all_inputs_as_internal(int);

TEST_F(NeuralnetworksHidlTest, tile_1_all_inputs_as_internal) {
  Execute(device,
          createTestModel_all_inputs_as_internal,
          is_ignored_all_inputs_as_internal,
          ::generated_tests::tile_1::get_examples_all_inputs_as_internal());
}

TEST_F(ValidationTest, tile_1_all_inputs_as_internal) {
  const Model model = createTestModel_all_inputs_as_internal();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_all_inputs_as_internal());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_all_inputs_as_internal_dynamic_output_shape();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_all_inputs_as_internal_dynamic_output_shape();
bool is_ignored_all_inputs_as_internal_dynamic_output_shape(int);

TEST_F(DynamicOutputShapeTest, tile_1_all_inputs_as_internal_dynamic_output_shape) {
  Execute(device,
          createTestModel_all_inputs_as_internal_dynamic_output_shape,
          is_ignored_all_inputs_as_internal_dynamic_output_shape,
          ::generated_tests::tile_1::get_examples_all_inputs_as_internal_dynamic_output_shape(), true);
}

TEST_F(ValidationTest, tile_1_all_inputs_as_internal_dynamic_output_shape) {
  const Model model = createTestModel_all_inputs_as_internal_dynamic_output_shape();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_all_inputs_as_internal_dynamic_output_shape());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_relaxed();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_relaxed();
bool is_ignored_relaxed(int);

TEST_F(NeuralnetworksHidlTest, tile_1_relaxed) {
  Execute(device,
          createTestModel_relaxed,
          is_ignored_relaxed,
          ::generated_tests::tile_1::get_examples_relaxed());
}

TEST_F(ValidationTest, tile_1_relaxed) {
  const Model model = createTestModel_relaxed();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_relaxed());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_relaxed_dynamic_output_shape();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_relaxed_dynamic_output_shape();
bool is_ignored_relaxed_dynamic_output_shape(int);

TEST_F(DynamicOutputShapeTest, tile_1_relaxed_dynamic_output_shape) {
  Execute(device,
          createTestModel_relaxed_dynamic_output_shape,
          is_ignored_relaxed_dynamic_output_shape,
          ::generated_tests::tile_1::get_examples_relaxed_dynamic_output_shape(), true);
}

TEST_F(ValidationTest, tile_1_relaxed_dynamic_output_shape) {
  const Model model = createTestModel_relaxed_dynamic_output_shape();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_relaxed_dynamic_output_shape());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_relaxed_all_inputs_as_internal();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_relaxed_all_inputs_as_internal();
bool is_ignored_relaxed_all_inputs_as_internal(int);

TEST_F(NeuralnetworksHidlTest, tile_1_relaxed_all_inputs_as_internal) {
  Execute(device,
          createTestModel_relaxed_all_inputs_as_internal,
          is_ignored_relaxed_all_inputs_as_internal,
          ::generated_tests::tile_1::get_examples_relaxed_all_inputs_as_internal());
}

TEST_F(ValidationTest, tile_1_relaxed_all_inputs_as_internal) {
  const Model model = createTestModel_relaxed_all_inputs_as_internal();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_relaxed_all_inputs_as_internal());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_relaxed_all_inputs_as_internal_dynamic_output_shape();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_relaxed_all_inputs_as_internal_dynamic_output_shape();
bool is_ignored_relaxed_all_inputs_as_internal_dynamic_output_shape(int);

TEST_F(DynamicOutputShapeTest, tile_1_relaxed_all_inputs_as_internal_dynamic_output_shape) {
  Execute(device,
          createTestModel_relaxed_all_inputs_as_internal_dynamic_output_shape,
          is_ignored_relaxed_all_inputs_as_internal_dynamic_output_shape,
          ::generated_tests::tile_1::get_examples_relaxed_all_inputs_as_internal_dynamic_output_shape(), true);
}

TEST_F(ValidationTest, tile_1_relaxed_all_inputs_as_internal_dynamic_output_shape) {
  const Model model = createTestModel_relaxed_all_inputs_as_internal_dynamic_output_shape();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_relaxed_all_inputs_as_internal_dynamic_output_shape());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_float16();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_float16();
bool is_ignored_float16(int);

TEST_F(NeuralnetworksHidlTest, tile_1_float16) {
  Execute(device,
          createTestModel_float16,
          is_ignored_float16,
          ::generated_tests::tile_1::get_examples_float16());
}

TEST_F(ValidationTest, tile_1_float16) {
  const Model model = createTestModel_float16();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_float16());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_float16_dynamic_output_shape();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_float16_dynamic_output_shape();
bool is_ignored_float16_dynamic_output_shape(int);

TEST_F(DynamicOutputShapeTest, tile_1_float16_dynamic_output_shape) {
  Execute(device,
          createTestModel_float16_dynamic_output_shape,
          is_ignored_float16_dynamic_output_shape,
          ::generated_tests::tile_1::get_examples_float16_dynamic_output_shape(), true);
}

TEST_F(ValidationTest, tile_1_float16_dynamic_output_shape) {
  const Model model = createTestModel_float16_dynamic_output_shape();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_float16_dynamic_output_shape());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_float16_all_inputs_as_internal();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_float16_all_inputs_as_internal();
bool is_ignored_float16_all_inputs_as_internal(int);

TEST_F(NeuralnetworksHidlTest, tile_1_float16_all_inputs_as_internal) {
  Execute(device,
          createTestModel_float16_all_inputs_as_internal,
          is_ignored_float16_all_inputs_as_internal,
          ::generated_tests::tile_1::get_examples_float16_all_inputs_as_internal());
}

TEST_F(ValidationTest, tile_1_float16_all_inputs_as_internal) {
  const Model model = createTestModel_float16_all_inputs_as_internal();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_float16_all_inputs_as_internal());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_float16_all_inputs_as_internal_dynamic_output_shape();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_float16_all_inputs_as_internal_dynamic_output_shape();
bool is_ignored_float16_all_inputs_as_internal_dynamic_output_shape(int);

TEST_F(DynamicOutputShapeTest, tile_1_float16_all_inputs_as_internal_dynamic_output_shape) {
  Execute(device,
          createTestModel_float16_all_inputs_as_internal_dynamic_output_shape,
          is_ignored_float16_all_inputs_as_internal_dynamic_output_shape,
          ::generated_tests::tile_1::get_examples_float16_all_inputs_as_internal_dynamic_output_shape(), true);
}

TEST_F(ValidationTest, tile_1_float16_all_inputs_as_internal_dynamic_output_shape) {
  const Model model = createTestModel_float16_all_inputs_as_internal_dynamic_output_shape();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_float16_all_inputs_as_internal_dynamic_output_shape());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_quant8();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_quant8();
bool is_ignored_quant8(int);

TEST_F(NeuralnetworksHidlTest, tile_1_quant8) {
  Execute(device,
          createTestModel_quant8,
          is_ignored_quant8,
          ::generated_tests::tile_1::get_examples_quant8());
}

TEST_F(ValidationTest, tile_1_quant8) {
  const Model model = createTestModel_quant8();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_quant8());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_quant8_dynamic_output_shape();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_quant8_dynamic_output_shape();
bool is_ignored_quant8_dynamic_output_shape(int);

TEST_F(DynamicOutputShapeTest, tile_1_quant8_dynamic_output_shape) {
  Execute(device,
          createTestModel_quant8_dynamic_output_shape,
          is_ignored_quant8_dynamic_output_shape,
          ::generated_tests::tile_1::get_examples_quant8_dynamic_output_shape(), true);
}

TEST_F(ValidationTest, tile_1_quant8_dynamic_output_shape) {
  const Model model = createTestModel_quant8_dynamic_output_shape();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_quant8_dynamic_output_shape());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_quant8_all_inputs_as_internal();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_quant8_all_inputs_as_internal();
bool is_ignored_quant8_all_inputs_as_internal(int);

TEST_F(NeuralnetworksHidlTest, tile_1_quant8_all_inputs_as_internal) {
  Execute(device,
          createTestModel_quant8_all_inputs_as_internal,
          is_ignored_quant8_all_inputs_as_internal,
          ::generated_tests::tile_1::get_examples_quant8_all_inputs_as_internal());
}

TEST_F(ValidationTest, tile_1_quant8_all_inputs_as_internal) {
  const Model model = createTestModel_quant8_all_inputs_as_internal();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_quant8_all_inputs_as_internal());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

namespace generated_tests::tile_1 {

std::vector<::test_helper::MixedTypedExample>& get_examples_quant8_all_inputs_as_internal_dynamic_output_shape();

} // namespace generated_tests::tile_1

namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1 {

Model createTestModel_quant8_all_inputs_as_internal_dynamic_output_shape();
bool is_ignored_quant8_all_inputs_as_internal_dynamic_output_shape(int);

TEST_F(DynamicOutputShapeTest, tile_1_quant8_all_inputs_as_internal_dynamic_output_shape) {
  Execute(device,
          createTestModel_quant8_all_inputs_as_internal_dynamic_output_shape,
          is_ignored_quant8_all_inputs_as_internal_dynamic_output_shape,
          ::generated_tests::tile_1::get_examples_quant8_all_inputs_as_internal_dynamic_output_shape(), true);
}

TEST_F(ValidationTest, tile_1_quant8_all_inputs_as_internal_dynamic_output_shape) {
  const Model model = createTestModel_quant8_all_inputs_as_internal_dynamic_output_shape();
  const std::vector<Request> requests = createRequests(::generated_tests::tile_1::get_examples_quant8_all_inputs_as_internal_dynamic_output_shape());
  validateEverything(model, requests);
}

} // namespace android::hardware::neuralnetworks::V1_2::generated_tests::tile_1

// clang-format off
// Generated file (from: strided_slice_float_11_relaxed.mod.py). Do not edit
#include "../../TestGenerated.h"

namespace strided_slice_float_11_relaxed {
// Generated strided_slice_float_11_relaxed test
#include "generated/examples/strided_slice_float_11_relaxed.example.cpp"
// Generated model constructor
#include "generated/models/strided_slice_float_11_relaxed.model.cpp"
} // namespace strided_slice_float_11_relaxed

TEST_F(GeneratedTests, strided_slice_float_11_relaxed) {
    execute(strided_slice_float_11_relaxed::CreateModel,
            strided_slice_float_11_relaxed::is_ignored,
            strided_slice_float_11_relaxed::get_examples());
}

#if 0
TEST_F(DynamicOutputShapeTests, strided_slice_float_11_relaxed_dynamic_output_shape) {
    execute(strided_slice_float_11_relaxed::CreateModel_dynamic_output_shape,
            strided_slice_float_11_relaxed::is_ignored_dynamic_output_shape,
            strided_slice_float_11_relaxed::get_examples_dynamic_output_shape());
}

#endif

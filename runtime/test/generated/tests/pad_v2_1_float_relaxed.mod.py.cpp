// clang-format off
// Generated file (from: pad_v2_1_float_relaxed.mod.py). Do not edit
#include "../../TestGenerated.h"

namespace pad_v2_1_float_relaxed {
// Generated pad_v2_1_float_relaxed test
#include "generated/examples/pad_v2_1_float_relaxed.example.cpp"
// Generated model constructor
#include "generated/models/pad_v2_1_float_relaxed.model.cpp"
} // namespace pad_v2_1_float_relaxed

TEST_F(GeneratedTests, pad_v2_1_float_relaxed) {
    execute(pad_v2_1_float_relaxed::CreateModel,
            pad_v2_1_float_relaxed::is_ignored,
            pad_v2_1_float_relaxed::get_examples());
}

#if 0
TEST_F(DynamicOutputShapeTests, pad_v2_1_float_relaxed_dynamic_output_shape) {
    execute(pad_v2_1_float_relaxed::CreateModel_dynamic_output_shape,
            pad_v2_1_float_relaxed::is_ignored_dynamic_output_shape,
            pad_v2_1_float_relaxed::get_examples_dynamic_output_shape());
}

#endif

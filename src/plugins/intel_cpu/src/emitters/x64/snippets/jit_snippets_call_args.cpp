// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_snippets_call_args.hpp"

namespace ov {
namespace intel_cpu {

#define GET_OFF(field) offsetof(jit_snippets_call_args, field)
#define GET_OFF_LOOP_ARGS(field) offsetof(jit_snippets_call_args::loop_args_t, field)

jit_snippets_call_args::loop_args_t::loop_args_t(int64_t work_amount, const std::vector<int64_t>& ptr_increments,
                                                         const std::vector<int64_t>& finalization_offsets)
    : m_work_amount(work_amount) {
    OPENVINO_ASSERT(ptr_increments.size() == finalization_offsets.size(),
                    "Inconsistent sizes of ptr_increments and finalization_offsets");
    m_num_data_ptrs = static_cast<int64_t>(ptr_increments.size());
    init_pointers_and_copy_data(m_num_data_ptrs, ptr_increments.data(), finalization_offsets.data());
}

jit_snippets_call_args::loop_args_t::loop_args_t(const loop_args_t& other)
    : m_work_amount(other.m_work_amount), m_num_data_ptrs(other.m_num_data_ptrs) {
    init_pointers_and_copy_data(m_num_data_ptrs, other.m_ptr_increments, other.m_finalization_offsets);
}

jit_snippets_call_args::loop_args_t::~loop_args_t() {
    delete[] m_ptr_increments;
    delete[] m_finalization_offsets;
}

jit_snippets_call_args::loop_args_t& jit_snippets_call_args::loop_args_t::operator=(loop_args_t other) {
    swap(*this, other);
    return *this;
}

jit_snippets_call_args::~jit_snippets_call_args() {
    delete[] loop_args;
}

void jit_snippets_call_args::register_loops(const std::vector<loop_args_t>& loops) {
    num_loops = loops.size();
    loop_args = new loop_args_t[num_loops];
    std::copy(loops.begin(), loops.end(), loop_args);
}

}   // namespace intel_cpu
}   // namespace ov

// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cstdint>
#include <cstring>

#include "openvino/core/except.hpp"

#include "constants.hpp"

namespace ov {
namespace intel_cpu {


#define GET_OFF(field) offsetof(jit_snippets_call_args, field)
#define GET_OFF_LOOP_ARGS(field) offsetof(jit_snippets_call_args::loop_args_t, field)

struct jit_snippets_call_args {
    struct loop_args_t;

    jit_snippets_call_args() = default;
    ~jit_snippets_call_args();

    void register_loops(const std::vector<loop_args_t>& loops);

    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *buffer_scratchpad_ptr = nullptr;

    // Note: Ideally loop_args must be private, since we manage this pointer manually.
    // However, standard-layout class definition (to use offset_of) requires the same access specifier
    // for all non-static data members. So we can keep them public or friend all control-flow emitters
    int32_t num_loops = 0;
    loop_args_t* loop_args = nullptr;
};

struct jit_snippets_call_args::loop_args_t {
    friend class jit_loop_begin_dynamic_emitter;
    friend class jit_loop_end_dynamic_emitter;

    loop_args_t() = default;
    loop_args_t(int64_t work_amount, const std::vector<int64_t>& ptr_increments, const std::vector<int64_t>& finalization_offsets);
    loop_args_t(const loop_args_t& other);
    ~loop_args_t();

    loop_args_t& operator=(loop_args_t other);
    friend void swap(loop_args_t& first, loop_args_t& second) {
        std::swap(first.m_work_amount, second.m_work_amount);
        std::swap(first.m_num_data_ptrs, second.m_num_data_ptrs);
        std::swap(first.m_ptr_increments, second.m_ptr_increments);
        std::swap(first.m_finalization_offsets, second.m_finalization_offsets);
    }

private:
    void init_pointers_and_copy_data(const int64_t num_elements,
                                     const int64_t* ptr_increments,
                                     const int64_t* finalization_offsets) {
        const size_t chunk_size = num_elements * sizeof(int64_t);
        m_ptr_increments = new int64_t[num_elements];
        std::memcpy(m_ptr_increments, ptr_increments, chunk_size);
        m_finalization_offsets = new int64_t[num_elements];
        std::memcpy(m_finalization_offsets, finalization_offsets, chunk_size);
    }

    //todo: can we use smaller data types?
    int64_t m_work_amount = 0;
    int64_t m_num_data_ptrs = 0;
    int64_t* m_ptr_increments = nullptr;
    int64_t* m_finalization_offsets = nullptr;
};

struct jit_snippets_compile_args {
    size_t parallel_executor_ndims = 1;
};

}   // namespace intel_cpu
}   // namespace ov

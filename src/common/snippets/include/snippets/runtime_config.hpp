// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/shape_inference/shape_inference.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {

/**
 * @interface RuntimeConfig
 * @brief Contain the runtime-dependent (shape-dependent) information: Loop parameters, offsets of data, buffer size
 */
class RuntimeConfig {
public:
    /**
     * @interface LoopDescriptor
     * @brief Describes Loops - the simple copy of LoopManager::LoopInfo without loop ports
     */
    struct LoopDescriptor {
        enum Type { Vector, Tile, SplitedTile };
        LoopDescriptor() = default;
        LoopDescriptor(size_t wa, size_t inc, std::vector<int64_t> ptr_incs = {}, std::vector<int64_t> final_offs = {}, Type type = Type::Vector)
            : work_amount(wa), increment(inc), ptr_increments(ptr_incs), finalization_offsets(final_offs), type(type) {}

        size_t work_amount = IShapeInferSnippets::DYNAMIC_DIMENSION;
        size_t increment = 1;
        std::vector<int64_t> ptr_increments = {};  // in bytes
        std::vector<int64_t> finalization_offsets = {};  // in bytes
        Type type = Type::Vector;
    };
    // [loop_id -> loop descriptors]
    using LoopMap = std::map<size_t, std::vector<LoopDescriptor>>;

    /**
     * @brief Find the LoopDescriptor and its ordered index by Loop ID and Type of descriptor
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @param desc the reference of the target loop descriptor
     * @param index the reference of ordered index of loop in all loop descriptors
     * @return True if the loop descriptor has been found. Otherwise returns False
     */
    bool get_loop_desc(size_t loop_id, LoopDescriptor::Type type, LoopDescriptor& desc, size_t& index) const;
    /**
     * @brief Find the LoopDescriptor by Loop ID and Type of descriptor.
     *        Since the method doesn't return ordered index of LoopDescriptor, this method is faster than previous.
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @param desc the reference of the target loop descriptor
     * @return True if the loop descriptor has been found. Otherwise returns False
     */
    bool get_loop_desc(size_t loop_id, LoopDescriptor::Type type, LoopDescriptor& desc) const;

    /**
     * @brief Return the loop descriptors
     * @return the const ref of the map [loop_id -> loop descriptors]
     */
    const LoopMap& get_loops() const { return loops; }
    /**
     * @brief Return the Subgraph input and output data offsets
     * @return the const ref of vector with data offsets
     */
    const std::vector<std::vector<int64_t>>& get_data_offsets() const { return data_offsets; }

    /**
     * @brief Initialize config using LinearIR state
     * @param tlinear_ir the updated LinearIR
     */
    void update(const lowered::LinearIR& linear_ir);

private:
    using LinearIR = lowered::LinearIR;
    /**
     * @brief Initialize the map of loops descriptors using LoopManager of LinearIR
     * @param loop_manager LoopManager of needed LinearIR
     */
    void init_loop_descriptors(const LinearIR::LoopManagerPtr& loop_manager);
    /**
     * @brief Optimize loop data shifts if loop evaluate once
     */
    void optimize_single_evaluation();
    /**
     * @brief Initialize the vector loop descriptor
     * @param loop_info loop information of the corresponding loop
     * @param vector_loop_desc ref of the vector loop descriptor which should be inited
     * @return True if the descriptor has been inited otherwise returns False
     */
    bool get_vector_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, LoopDescriptor& vector_loop_desc);
    /**
     * @brief Initialize the tail loop descriptor
     * @param loop_info loop information of the corresponding loop
     * @param vector_loop_desc ref of the vector loop descriptor which may be updated after splitting to vector and tail loops
     * @param tail_loop_desc ref of the tail loop descriptor which should be inited
     * @return True if the descriptor has been inited otherwise returns False
     */
    bool get_tail_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, LoopDescriptor& vector_loop_desc, LoopDescriptor& tail_loop_desc);
    /**
     * @brief Initialize the inner tail splited loops
     * @param loop_manager LoopManager of needed LinearIR
     * @param outer_splited_loop_info loop information of the outer splited loop
     * @param outer_splited_tail_loop_desc tail descriptor of the outer splited loop
     * @param outer_loop_id ID of the outer splited loop
     * @param is_outer_vector_loop_needed bool value if outer splited loop has vector loop
     */
    void init_inner_splited_tail_loop_descriptors(const LinearIR::LoopManagerPtr& loop_manager,
                                                  const LinearIR::LoopManager::LoopInfoPtr& outer_splited_loop_info,
                                                  const LoopDescriptor& outer_splited_tail_loop_desc,
                                                  size_t outer_loop_id, bool is_outer_vector_loop_needed);
    /**
     * @brief Check if vector loop is needed
     * @param loop_info loop information of the corresponding loop
     * @return True if needed otherwise returns False
     */
    inline static bool is_vector_loop_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
        return loop_info->work_amount >= loop_info->increment;
    }
    /**
     * @brief Check if tail loop is needed
     * @param loop_info loop information of the corresponding loop
     * @return True if needed otherwise returns False
     */
    inline static bool is_tail_loop_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
        return loop_info->work_amount % loop_info->increment != 0;
    }

    // [loop_id -> loop descriptors]
    LoopMap loops;
    // offsets of subgraph input and output data
    std::vector<std::vector<int64_t>> data_offsets;
};

} // namespace snippets
} // namespace ov

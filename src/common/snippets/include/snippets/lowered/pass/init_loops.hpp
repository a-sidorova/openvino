// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformation.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InitLoops
 * @brief The pass explicitly insert LoadBegin and LoadEnd in Linear IR using Loop markup
 * @ingroup snippets
 */
class InitLoops : public Transformation {
public:
    OPENVINO_RTTI("InsertLoops", "Transformation")
    InitLoops();
    bool run(LinearIR& linear_ir) override;

private:
    bool insertion(LinearIR& linear_ir, const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                   size_t loop_id, size_t dim_idx, bool has_outer_loop);
    std::vector<int64_t> init_ptr_increments(const std::vector<TensorDescriptor>& loop_inputs,
                                             const std::vector<TensorDescriptor>& loop_outputs,
                                             size_t dim_idx) const;
    std::vector<int64_t> init_finalization_offsets(const std::vector<int64_t>& finalization_offsets, size_t work_amount) const;
    std::vector<int64_t> init_element_type_sizes(const std::vector<TensorDescriptor>& loop_inputs,
                                                 const std::vector<TensorDescriptor>& loop_outputs);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph

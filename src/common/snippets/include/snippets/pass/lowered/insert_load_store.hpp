// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface InsertLoadStore
 * @brief InsertLoadStore
 * @ingroup snippets
 */
class InsertLoadStore : public LinearIRTransformation {
public:
    explicit InsertLoadStore(size_t vector_size);
    OPENVINO_RTTI("InsertLoadStore", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;

private:
    void insert_load(LoweredExprIR& linear_ir, const LoweredLoopManagerPtr& loop_manager, const LoweredExprPort& entry_point,
                     LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos);
    void insert_store(LoweredExprIR& linear_ir,  const LoweredLoopManagerPtr& loop_manager, const LoweredExprPort& exit_point,
                      LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos);
    void update_loops(const LoweredLoopManagerPtr& loop_manager, const std::vector<size_t>& loop_ids,
                      const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry = true);
    void update_loop(const LoweredLoopManager::LoweredLoopInfoPtr& loop_info,
                     const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry = true);
    std::vector<size_t> get_loops_to_update(const std::vector<size_t>& loop_ids, size_t loop_id);

    size_t m_vector_size;
};

} //namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

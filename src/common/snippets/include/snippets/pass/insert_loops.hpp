// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface InsertLoops
 * @brief Insert explicit Loop operations into the body to process multiple data entities during one kernel execution
 * @param master_shape - shape used to determine loop work amounts
 * @param loop_depth - the number of last master_shape dimensions processed by loops (aka tileRank - obsolete), could be 1 or 2
 * @param vector_size - the number of entities processed on one iteration of vector loop
 * @ingroup snippets
 */
class InsertLoops: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("InsertLoops", "0");
    InsertLoops(ov::PartialShape master_shape, size_t loop_depth, size_t vector_size, bool is_optimized = true);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

    static std::vector<bool> calculate_inner_apply_increments(const ov::PartialShape& master, const std::vector<ov::PartialShape>& shapes);
    static std::vector<bool> calculate_outer_apply_increments(const std::vector<ov::PartialShape>& shapes);
    static std::vector<int64_t> calculate_finalization_offsets(const ov::PartialShape& master, const std::vector<ov::PartialShape>& shapes);
private:
    ov::PartialShape m_master_shape;
    size_t m_loop_depth;
    size_t m_vector_size;
    bool m_is_optimized;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

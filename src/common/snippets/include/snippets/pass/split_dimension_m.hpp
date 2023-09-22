// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface SplitDimensionM
 * @brief Inserts Reshape nodes after and before Parameters and Results in Subgraphs with MatMul inside
 *        to split dimension M for MatMuls to increase work amount for parallelism
 *        Note: works only with 3D MHA patterns
 * @ingroup snippets
 */
class SplitDimensionM: public CommonOptimizations::SubgraphPass {
public:
    OPENVINO_RTTI("SplitDimensionM", "0");
    SplitDimensionM(size_t concurrency) : m_concurrency(concurrency) {}

    bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) override;

    // Return True if the MatMul node is supported by this optimization
    static bool is_supported_matmul(const std::shared_ptr<const ov::Node>& node);
    // Returns True if parallelism work amount (concurrency) can be increased by this optimization
    static bool can_be_optimized(const std::shared_ptr<const ov::Node>& node, size_t concurrency);

private:
    static std::shared_ptr<ov::op::v0::MatMul> get_matmul(const std::shared_ptr<op::Subgraph>& subgraph);
    bool get_optimized_dimensions(const ov::Shape& shape, size_t& batch_m_dim, size_t& new_m_dim) const;

    void reshape_subgraph(const std::shared_ptr<op::Subgraph>& subgraph, const ov::Shape& shape, size_t batch_m_dim, size_t new_m_dim);

    size_t m_concurrency;
    static constexpr float m_optimal_thread_num_percent = 0.8;
};


} // namespace pass
} // namespace snippets
} // namespace ov

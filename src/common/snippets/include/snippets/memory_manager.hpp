// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public interface for memory manager.
 * @file memory_manager.hpp
 */
#pragma once

#include "snippets_isa.hpp"

#include "memory_solver.hpp"
#include "pass/tokenization.hpp"

namespace ngraph {
namespace snippets {

using BufferCluster = std::set<std::shared_ptr<op::Buffer>>;
using BufferClusters = std::vector<BufferCluster>;

/**
 * @interface MemoryManager
 * @brief Helps to solve issue of optimal memory allocation only for Buffers in graph using MemorySolver
 * @ingroup snippets
 */
class MemoryManager {
public:
    MemoryManager(const std::shared_ptr<ov::Model>& model);

    /**
     * @brief allocate optimal memory size using MemorySolver
     * @return size of common memory blob
     */
    int64_t allocate() const;

private:
    /**
     * @brief init Buffers as graph edges and other subgraph around the Buffers as Nodes using enumeration
     *        Parameter
     *    |--- LoopBegin         Parameter
     *    |   LoadReshape         <Edge>   <- already allocated. Skip
     *    |     Store      --->    Node    <- (LoopBegin,...,LoopEnd)
     *    |--- LoopEnd            <Edge>   <- Buffer. Intermediate memory (edge)
     *          Buffer             Node    <- (LoopBegin,...,LoopEnd)
     *    |--- LoopBegin           ...
     *    |      ...
     */
    void init_edges(const std::shared_ptr<ov::Model>& model);

    /**
     * @brief init boxes for MemorySolver
     */
    void init_boxes();

    /**
     * @brief set offsets to Buffer and propagate to the previous and the next nodes
     */
    void set_offset(const std::shared_ptr<op::Buffer>& buffer, const size_t offset) const;


    BufferClusters edge_clusters;
    std::vector<MemorySolver::Box> boxes;
    constexpr static int64_t alignment = 32;  // 32 bytes
};

} // namespace snippets
} // namespace ngraph

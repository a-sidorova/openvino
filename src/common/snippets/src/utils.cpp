// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils.hpp"

#include "snippets/pass/fq_decomposition.hpp"
#include "openvino/core/rt_info.hpp"


namespace ov {
namespace snippets {
namespace utils {

auto get_non_scalar_constant_count_for_fq(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) -> size_t {
    std::vector<float> cl, ch, isc, ish, osc, osh;
    const bool status = ov::snippets::pass::FakeQuantizeDecomposition::getScalesAndShifts(fq, cl, ch, isc, ish, osc, osh);
    bool is_optimized = false;  // The case when we can calculate only scales
    if (status) {
        const auto out_scales = ov::snippets::pass::FakeQuantizeDecomposition::calculateScales(fq->get_output_element_type(0), cl, ch, isc, ish, osc, osh);
        is_optimized = out_scales.size() != 0;
    }

    const bool only_quantized = is_optimized || (status &&
                                                 std::all_of(osc.cbegin(), osc.cend(),
                                                     [](float val) { return val == 1.f; }) &&
                                                 std::all_of(osh.cbegin(), osh.cend(),
                                                     [](float val) { return val == 0.f; }));
    const bool il = ov::shape_size(fq->input(1).get_shape()) != 1lu;
    const bool ih = ov::shape_size(fq->input(2).get_shape()) != 1lu;
    const bool ol = !only_quantized && ov::shape_size(fq->input(3).get_shape()) != 1lu;
    const bool oh = !only_quantized && ov::shape_size(fq->input(4).get_shape()) != 1lu;

    // FakeQuantize decompoisition has the folowwing formula:
    //      round(x * (levels-1) / (ih - il) - il * (levels-1) / (ih - il)) * (oh - ol) / (levels-1) + ol
    // After the decomposition there is call of ConstantsFolding pass that generates new Constants:
    //      - isc := (levels-1) / (ih - il)
    //      - ish := -il * isc
    //      - osc := (oh - ol) / (levels-1)
    //      - osh := ol
    // New formula:
    //      round(x * isc + ish) * osc + osh
    // Thus, after FakeQuantize decompoisition we have:
    //      - If it's non optimized FQ, 6 Constants instead of original 4:
    //              ih, il (for Max/Min), isc, ish, osc, osh
    //      - If it's optimized FQ, 3 Constants instead of original 4:
    //              ih, il (for Max/Min), isc
    // Some of them can be scalar or non-scalar. It depends on which original 4 Constants are non-scalar
    // To sum it up, below conditions check all possible cases to calculate count of new generated non-scalars
    if (is_optimized) {
        if (il && ih)
            return 3;
        else if (il || ih)
            return 2;
        return 0;
    } else {
        if (ol && il && ih)
            return 6;
        else if ((ol && (il || ih)) || (il && ih && oh))
            return 5;
        else if ((il && oh) || (ih && oh) || (il && ih))
            return 4;
        else if (il || ih)
            return 3;
        else if (ol)
            return 2;
        else if (oh)
            return 1;
        return 0;
    }
}

VectorDims get_planar_vdims(const VectorDims& shape, const std::vector<size_t>& order) {
    if (order.empty())
        return shape;
    VectorDims reordered_shape(order.size());
    ordered_vector(shape, order, true, reordered_shape);
    return reordered_shape;
}
VectorDims get_preordered_vdims(const VectorDims& shape, const std::vector<size_t>& order) {
    if (order.empty())
        return shape;
    VectorDims reordered_shape(order.size());
    ordered_vector(shape, order, false, reordered_shape);
    return reordered_shape;
}

VectorDims get_planar_vdims(const snippets::lowered::ExpressionPort& expr_port) {
    OPENVINO_ASSERT(expr_port.get_type() == snippets::lowered::ExpressionPort::Type::Input, "get_planar_vdims expects Expression Input port");
    if (const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr_port.get_expr()->get_node())) {
        return utils::get_planar_vdims(expr_port.get_descriptor_ptr()->get_shape(), ma->get_input_order(expr_port.get_index()));
    }
    return expr_port.get_descriptor_ptr()->get_shape();
}
VectorDims get_preordered_vdims(const snippets::lowered::ExpressionPort& expr_port) {
    OPENVINO_ASSERT(expr_port.get_type() == snippets::lowered::ExpressionPort::Type::Output, "get_preordered_vdims expects Expression Output port");
    if (const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr_port.get_expr()->get_node())) {
        return utils::get_preordered_vdims(expr_port.get_descriptor_ptr()->get_shape(), ma->get_output_order(expr_port.get_index()));
    }
    return expr_port.get_descriptor_ptr()->get_shape();
}

bool is_dynamic_vdims(const VectorDims& shape) {
    return std::any_of(shape.cbegin(), shape.cend(), [](size_t v){ return v == IShapeInferSnippets::DYNAMIC_DIMENSION; });
}

VectorDims pshape_to_vdims(const PartialShape& pshape) {
    VectorDims result;
    result.reserve(pshape.size());
    for (const auto& d : pshape)
        result.push_back(d.is_dynamic() ? IShapeInferSnippets::DYNAMIC_DIMENSION : d.get_length());
    // Note: PartialShape could be empty which designates scalar value. However, Scalars are represented as {1} in Snippets
    return result.empty() ? VectorDims {1} : result;
}

ov::PartialShape vdims_to_pshape(const VectorDims& vdims) {
    ov::PartialShape result;
    result.reserve(vdims.size());
    for (const auto& v : vdims)
        result.push_back(v != IShapeInferSnippets::DYNAMIC_DIMENSION ?
                         Dimension(static_cast<Dimension::value_type>(v)) :
                         Dimension());
    return result;
}

} // namespace utils
} // namespace snippets
} // namespace ov

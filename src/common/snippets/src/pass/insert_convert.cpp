// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/insert_convert.hpp"
#include "ngraph_ops/type_relaxed.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <snippets/snippets_isa.hpp>
#include <ngraph/pattern/op/or.hpp>

ngraph::snippets::pass::InsertConvertAfterLoadAndScalars::InsertConvertAfterLoadAndScalars(const ov::element::TypeVector& supported_exec_types) {
    MATCHER_SCOPE(InsertConvertAfterLoadAndScalars);

    ngraph::graph_rewrite_callback callback = [&](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertConvertAfterLoadAndScalars")
        auto root = m.get_match_root();

        auto is_supported_type = [supported_exec_types](const ov::element::Type& type) {
            return any_of(supported_exec_types.begin(), supported_exec_types.end(),
                          [&type](const ov::element::Type& supported_type) { return supported_type == type; } );
        };

        if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(root)) {
            // we shouldn't change existing Convert before Store
            for (auto consumer : convert->output(0).get_target_inputs()) {
                if (ov::is_type<snippets::op::Store>(consumer.get_node()->shared_from_this())) {
                    return false;
                }
            }

            // we shouldn't change existing Convert inside body (and after Load) with supported element type
            if (is_supported_type(convert->get_destination_type())) {
                return false;
            }

            const auto new_convert = std::make_shared<ov::op::v0::Convert>(convert->get_input_node_shared_ptr(0), supported_exec_types.front());
            replace_node(root, new_convert);
            return true;
        }

        bool rewritten = false;
        for (auto input : root->inputs()) {
            if (is_supported_type(input.get_element_type())) {
                continue;
            }
            const auto source = input.get_source_output().get_node()->shared_from_this();
            if (ov::is_type<snippets::op::Load>(source) ||
                ov::is_type<snippets::op::BroadcastLoad>(source) ||
                ov::is_type<snippets::op::Scalar>(source)) {
                const auto new_convert = std::make_shared<ngraph::opset1::Convert>(source,
                                                                                   supported_exec_types.front());
                input.replace_source_output(new_convert);
                rewritten |= true;
            }
        }

        return rewritten;
    };

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(pattern::any_input(), matcher_name), callback);
}

ngraph::snippets::pass::PrecisionPropagation::PrecisionPropagation(const ov::element::Type default_type) : default_type(default_type) { }

bool ngraph::snippets::pass::PrecisionPropagation::run_on_model(const std::shared_ptr<ov::Model> &m) {
    bool rewritten = false;
    for (auto& op : m->get_ops()) {
        auto node = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(op);
        if (!!node && !ov::is_type<ngraph::op::v0::Convert>(op)) {
            for (int i = 0; i < op->outputs().size(); i++) {
                node->set_overridden_output_type(default_type, i);
                rewritten |= true;
            }  // Convert { <type> -> u8 } can be decomposed on Round and Clamp
        }
    }

    return rewritten;
}

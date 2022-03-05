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

ngraph::snippets::pass::InsertConvertAfter::InsertConvertAfter(const ov::element::TypeVector& supported_exec_types) {
    MATCHER_SCOPE(InsertConvertAfter);

    const auto input_pattern = ngraph::pattern::any_input();
    auto constant_pattern = std::make_shared<pattern::op::Label>(pattern::any_input(),
                                                          [](std::shared_ptr<Node> n) {
                                                              return ngraph::is_type<snippets::op::Scalar>(n);
                                                          });
    const auto load_pattern = ngraph::pattern::wrap_type<snippets::op::Load>({input_pattern});
    const auto broadcast_load_pattern = ngraph::pattern::wrap_type<snippets::op::BroadcastLoad>({input_pattern});
    const auto or_pattern =
            std::make_shared<pattern::op::Or>(OutputVector{constant_pattern, broadcast_load_pattern, load_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertConvertAfter")
        const auto& pattern_map = m.get_pattern_value_map();
        std::shared_ptr<Node> node = nullptr;
        if (pattern_map.count(constant_pattern)) {
            node = pattern_map.at(constant_pattern).get_node_shared_ptr();
        } else if (pattern_map.count(load_pattern)) {
            node = pattern_map.at(load_pattern).get_node_shared_ptr();
        } else if (pattern_map.count(broadcast_load_pattern)) {
            node = pattern_map.at(broadcast_load_pattern).get_node_shared_ptr();
        } else {
            return false;
        }

        auto is_supported_type = [supported_exec_types](const ov::element::Type& type) {
            return any_of(supported_exec_types.begin(), supported_exec_types.end(),
                          [&type](const ov::element::Type& supported_type) { return supported_type == type; } );
        };

        const auto new_convert = std::make_shared<ngraph::opset1::Convert>(node, supported_exec_types.front());
        ngraph::copy_runtime_info(node, new_convert);

        bool rewritten = false;
        for (auto output : node->outputs()) {
            for (auto consumer : output.get_target_inputs()) {
                const auto& consumer_node = consumer.get_node()->shared_from_this();
                // change existing convert if it has incorrect destination type
                if (const auto& convert = ov::as_type_ptr<opset1::Convert>(consumer_node)) {
                    const auto output_type = convert->get_output_element_type(0);
                    if (!is_supported_type(output_type)) {
                        replace_node(convert, new_convert);
                        rewritten |= true;
                    }
                } else {
                    const auto output_type = consumer.get_element_type();
                    if (!is_supported_type(output_type)) {
                        consumer.replace_source_output(new_convert);
                        consumer_node->validate_and_infer_types();
                        rewritten |= true;
                    }
                }
            }
        }

        return rewritten;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(or_pattern, matcher_name);
    this->register_matcher(m, callback);
}


ngraph::snippets::pass::InsertConvertBeforeStore::InsertConvertBeforeStore(const ov::element::TypeVector& supported_exec_types) {
    MATCHER_SCOPE(InsertConvertBeforeStore);

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<snippets::op::Store>()),
                     [this](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertConvertBeforeStore")
        auto root = m.get_match_root();

        const auto output_type = root->get_element_type();

        bool rewritten = false;
        for (auto input : root->inputs()) {
            const auto source = input.get_source_output().get_node()->shared_from_this();
            // change existing convert if it has incorrect destination type
            if (const auto& convert = ov::as_type_ptr<opset1::Convert>(source)) {
                if (convert->get_destination_type() != output_type) {
                    const auto new_convert = std::make_shared<ngraph::opset1::Convert>(convert->get_input_node_shared_ptr(0), output_type);
                    replace_node(convert->get_input_node_shared_ptr(0), new_convert);
                    rewritten |= true;
                }
            } else {
                const auto new_convert = std::make_shared<ngraph::opset1::Convert>(source, output_type);
                input.replace_source_output(new_convert);
                rewritten |= true;
            }
        }

        return rewritten;
    });
}

ngraph::snippets::pass::PrecisionPropagation::PrecisionPropagation(const ov::element::Type default_type) : default_type(default_type) { }

bool ngraph::snippets::pass::PrecisionPropagation::run_on_model(const std::shared_ptr<ngraph::Function> &m) {
    bool rewritten = false;
    for (auto& op : m->get_ops()) {
        auto node = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(op);
        if (!!node && !ov::is_type<ngraph::op::v0::Convert>(op)) {
            for (int i = 0; i < op->outputs().size(); i++) {
                node->set_overridden_output_type(default_type, i);
                rewritten |= true;
            }
        }
    }

    return rewritten;
}

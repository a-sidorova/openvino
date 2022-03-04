// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/insert_convert.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <snippets/snippets_isa.hpp>
#include <ngraph/pattern/op/or.hpp>

ngraph::snippets::pass::InsertConvertAfterLoad::InsertConvertAfterLoad(const ov::element::TypeVector& supported_exec_types) {
    MATCHER_SCOPE(InsertConvert);

    const auto input_pattern = ngraph::pattern::any_input();
    const auto load_pattern = ngraph::pattern::wrap_type<snippets::op::Load>({input_pattern});
    const auto broadcast_load_pattern = ngraph::pattern::wrap_type<snippets::op::BroadcastLoad>({input_pattern});
    const auto load_or_broadcast_load_pattern =
            std::make_shared<pattern::op::Or>(OutputVector{load_pattern, broadcast_load_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        std::shared_ptr<Node> load_node = nullptr;
        if (pattern_map.count(load_pattern)) {
            const auto load = pattern_map.at(load_pattern);
            load_node = std::dynamic_pointer_cast<snippets::op::Load>(load.get_node_shared_ptr());
        } else if (pattern_map.count(broadcast_load_pattern)) {
            const auto broadcast_load = pattern_map.at(broadcast_load_pattern);
            load_node = std::dynamic_pointer_cast<snippets::op::BroadcastLoad>(broadcast_load.get_node_shared_ptr());
        } else {
            return false;
        }

        auto is_supported_type = [supported_exec_types](const ov::element::Type& type) {
            return any_of(supported_exec_types.begin(), supported_exec_types.end(),
                          [&type](const ov::element::Type& supported_type) { return supported_type == type; } );
        };

        const auto new_convert = std::make_shared<ngraph::opset1::Convert>(load_node, supported_exec_types.front());
        ngraph::copy_runtime_info(load_node, new_convert);

        bool rewritten = false;
        for (auto output : load_node->outputs()) {
            for (auto consumer : output.get_target_inputs()) {
                const auto& consumer_node = consumer.get_node()->shared_from_this();
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
                        rewritten |= true;
                    }
                }
            }
        }

        return rewritten;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(load_or_broadcast_load_pattern, matcher_name);
    this->register_matcher(m, callback);
}


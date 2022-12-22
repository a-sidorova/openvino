// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/propagate_precision.hpp"

#include <assert.h>
#include <memory>
#include "ov_ops/type_relaxed.hpp"
#include "snippets/itt.hpp"
#include "ngraph/rt_info.hpp"

using namespace ngraph;

ngraph::snippets::pass::PropagatePrecision::PropagatePrecision(
    const ov::element::Type supported_precision,
    const std::shared_ptr<const TargetMachine>& target_machine) : supported_precision(supported_precision), target_machine(target_machine) {
}

bool ngraph::snippets::pass::PropagatePrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(PropagatePrecision);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::PropagatePrecision")

    std::unordered_map<std::shared_ptr<ngraph::opset1::Result>, element::Type> result_types;
    auto results = f->get_results();
    for (auto& result : results) {
        result_types.emplace(result, result->get_input_source_output(0).get_element_type());
    }

    for (const auto& op : f->get_ordered_ops()) {
        if (ngraph::is_type<opset1::Constant>(op)) {
            continue;
        }

        auto type_info = op->get_type_info();
        if (!target_machine->has(type_info)) {
            throw ov::Exception(
                "operation '" + std::string(type_info.version_id) + "::" + std::string(type_info.name) +
                "' was not found in target machine");
        }

        auto exec = target_machine->get_supported_precisions(type_info);
        const auto supported_precisions = exec(op);
        if (supported_precisions.empty()) {
            continue;
        }

        bool alligned_inputs = true;
        for (const auto& input : op->inputs()) {
            if (!ov::is_type<ngraph::snippets::op::ConvertSaturation>(input.get_source_output().get_node())) {
                alligned_inputs = false;
                break;
            }
        }

        std::vector<element::Type> input_precisions;
        for (const auto& input : op->inputs()) {
            const auto parent_input = alligned_inputs ? input.get_source_output().get_node()->input(0) : input;

            const auto input_precision = parent_input.get_source_output().get_element_type();
            input_precisions.push_back(input_precision);
        }

        assert(std::all_of(
            supported_precisions.begin(),
            supported_precisions.end(),
            [&input_precisions](const std::vector<element::Type>& precisions) {
                return precisions.size() == input_precisions.size();
            }) && "input precisions count is not equal for supported precisions");

        auto type_relaxed_node = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(op);

        // if possible remove alligned input convertions
        if (alligned_inputs &&
            std::any_of(
                supported_precisions.begin(),
                supported_precisions.end(),
                [&input_precisions](const std::vector<element::Type>& precisions) {
                    return precisions == input_precisions;
                })) {
            std::vector<element::Type> original_types;
            for (const auto& output : op->outputs()) {
                original_types.push_back(output.get_element_type());
            }

            for (auto i = 0ull; i < op->get_input_size(); ++i) {
                const auto convert = op->get_input_node_shared_ptr(i);
                assert(ov::is_type<ngraph::snippets::op::ConvertSaturation>(convert));
                op->set_argument(i, convert->input(0).get_source_output());
            }

            op->validate_and_infer_types();

            auto insert_final_convert = false;
            for (auto i = 0ull; i < op->get_output_size(); ++i) {
                if (original_types[i] != op->output(i).get_element_type()) {
                    insert_final_convert = true;
                    break;
                }
            }

            if (insert_final_convert) {
                for (auto i = 0ull; i < op->get_output_size(); ++i) {
                    const auto& op_output = op->output(i);
                    for (const auto& input : op_output.get_target_inputs()) {
                        const auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                            op_output,
                            original_types[i]);
                        input.replace_source_output(convert->output(0));
                    }
                }
            }

            if (type_relaxed_node == nullptr) {
                continue;
            }
        }

        auto input_precisions_were_changed = false;

        // update input precisions
        // if possible then convert precisions to supported
        if (!supported_precisions.empty() &&
            !std::any_of(
                supported_precisions.begin(),
                supported_precisions.end(),
                [&input_precisions](const std::vector<element::Type>& precisions) {
                    return precisions == input_precisions;
                })) {
            auto precisions = get_precisions(input_precisions,
                                             supported_precisions,
                                             supported_precision);
            if (precisions.empty()) {
                throw ov::Exception(
                    "there are no supported precisions for operation '" +
                    std::string(type_info.version_id) + "::" +
                    std::string(type_info.name) + "'");
            }

            for (auto i = 0ull; i < op->get_input_size(); ++i) {
                const auto& op_input = op->input(i);
                const auto& required_after = precisions[i];
                auto parent_output = op_input.get_source_output();
                const auto actual_before = parent_output.get_element_type();
                if (actual_before != required_after) {
                    input_precisions_were_changed = true;
                    auto existing_convert = ngraph::as_type<ngraph::snippets::op::ConvertSaturation>(
                        parent_output.get_node());
                    if (existing_convert == nullptr) {
                        auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                            parent_output,
                            required_after);
                        ngraph::copy_runtime_info(parent_output.get_node_shared_ptr(), convert);
                        op->set_argument(op_input.get_index(), convert);
                    } else {
                        const auto actual_before = existing_convert->get_input_source_output(0).get_element_type();
                        const auto actual_after = existing_convert->output(0).get_element_type();
                        if (can_be_removed(actual_before, actual_after, required_after)) {
                            existing_convert->output(0).replace(parent_output);
                        } else {
                            if (can_be_fused(actual_after, required_after)) {
                                auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                                    existing_convert->get_input_node_shared_ptr(0),
                                    required_after);
                                ngraph::copy_runtime_info(parent_output.get_node_shared_ptr(), convert);
                                op->set_argument(op_input.get_index(), convert);
                            } else {
                                auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                                    existing_convert->output(0),
                                    required_after);
                                ngraph::copy_runtime_info(existing_convert->shared_from_this(), convert);
                                op->set_argument(op_input.get_index(), convert);
                            }
                        }
                    }
                }
            }
        }

        if (input_precisions_were_changed || (type_relaxed_node != nullptr)) {
            // update output precision
            std::vector<element::Type> op_output_types;
            for (auto& output : op->outputs()) {
                op_output_types.push_back(output.get_element_type());
            }

            if (type_relaxed_node != nullptr) {
                // to keep previous functionality
                // unary and binary element-wise operations are supported
                // will be replaced to snippets opset later
                const auto op_element_type = op->get_input_source_output(0).get_element_type();
                if (type_relaxed_node->get_overridden_output_type(0) != op_element_type) {
                    assert(op->get_output_size() == 1ull);

                    type_relaxed_node->set_overridden_output_type(op_element_type, 0);
                    std::dynamic_pointer_cast<ngraph::Node>(op)->validate_and_infer_types();
                }
            } else {
                op->validate_and_infer_types();
            }

            for (auto i = 0ull; i < op->get_output_size(); ++i) {
                const auto& output = op->output(i);
                if (output.get_element_type() != op_output_types[i]) {
                    auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                        output,
                        op_output_types[i]);
                    ngraph::copy_runtime_info(output.get_node_shared_ptr(), convert);
                    // TODO: add tests: we should insert one conversion before several consumers
                    for (auto& input : output.get_target_inputs()) {
                        auto child = input.get_node();
                        if (child == convert.get()) {
                            continue;
                        }
                        input.replace_source_output(convert->output(0));
                    }
                }
            }
        }
    }

    for (auto it = result_types.begin(); it != result_types.end(); ++it) {
        const auto result = it->first;
        const auto actual_type = result->get_input_source_output(0).get_element_type();
        const auto expected_type = it->second;
        if (actual_type != it->second) {
            auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                result->get_input_node_shared_ptr(0),
                expected_type);
            ngraph::copy_runtime_info(result->get_input_node_shared_ptr(0), convert);
            result->set_argument(0, convert);
        }
    }

    return false;
}

bool ngraph::snippets::pass::PropagatePrecision::can_be_removed(
    const element::Type& actual_before,
    const element::Type& actual_after,
    const element::Type& required_after) {
    if (actual_before != required_after) {
        return false;
    }

    // TODO: just as example, have to be generalized
    if ((actual_before == element::u8) && (actual_after == element::f32)) {
        return true;
    }

    return false;
}

bool ngraph::snippets::pass::PropagatePrecision::can_be_fused(
    const element::Type& actual_after,
    const element::Type& required_after) {
    return
        (actual_after.is_real() == required_after.is_real()) &&
        (actual_after.bitwidth() >= required_after.bitwidth());
}

std::vector<element::Type> ngraph::snippets::pass::PropagatePrecision::get_precisions(
    const std::vector<element::Type>& input_precisions,
    const std::set<std::vector<element::Type>>& supported_precisions_pack,
    const element::Type& base_precision) noexcept {
    bool was_found = false;
    for (const auto& supported_precisions : supported_precisions_pack) {
        for (auto i = 0ull; i < supported_precisions.size(); ++i) {
            const auto& supported_precision = supported_precisions[i];
            const auto& input_precision = input_precisions[i];
            if ((supported_precision.is_real() != input_precision.is_real()) ||
                (input_precision.bitwidth() > supported_precision.bitwidth())) {
                was_found = false;
                break;
            }

            was_found = true;
        }
        if (was_found) {
            return supported_precisions;
        }
    }
    for (const auto& supported_precisions : supported_precisions_pack) {
        if (supported_precisions[0] == base_precision) {
            return supported_precisions;
        }
    }
    return {};
}

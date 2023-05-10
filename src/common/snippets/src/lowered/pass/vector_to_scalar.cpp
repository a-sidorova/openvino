// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/vector_to_scalar.hpp"

#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

SetScalarCountForLoadStore::SetScalarCountForLoadStore() {}

bool SetScalarCountForLoadStore::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::SetScalarCountForLoadStore")
    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& op = expr_it->get()->get_node();
        const auto load = ov::as_type_ptr<op::Load>(op);
        const auto store = ov::as_type_ptr<op::Store>(op);
        if (load || store) {
            const auto& tensor = load ? (*expr_it)->get_input_tensor(0)
                                      : (*expr_it)->get_output_tensor(0);
            const auto& layout = tensor->get_layout();
            const auto& tensor_shape = tensor->get_shape();
            // Find last dimension by layout
            const auto last_dim_idx = std::find(layout.begin(), layout.end(), layout.size() - 1);
            OPENVINO_ASSERT(last_dim_idx != layout.end(), "Load/Store expression have incorrect layout");
            const auto dim = tensor_shape[*last_dim_idx];
            if (dim == 1) {
                modified |= true;
                if (load) load->set_count(1lu);
                if (store) store->set_count(1lu);
            }
        }
    }
    return modified;
}



} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph

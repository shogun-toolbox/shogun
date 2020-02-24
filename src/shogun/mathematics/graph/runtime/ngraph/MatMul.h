/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ADD_NGRAPH_H_
#define SHOGUN_ADD_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

#include <ngraph/op/fused/matmul.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class AddNGraph

				    : public RuntimeNodeTemplate<node::MatMul, ::ngraph::Node>
				{
				public:
					AddNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Add";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						const auto input_node =
						    std::static_pointer_cast<node::MatMul>(node);
						if (input_node.size() != 2)
							error("Expected two input nodes in "
							      "AddNGraph.");
						const bool transpose_a = input_node->get_transpose_a();
						const bool transpose_b = input_node->get_transpose_b();

						return std::make_shared<::ngraph::op::MatMul>(
						    input_node[0], input_node[1], transpose_a,
						    transpose_b);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

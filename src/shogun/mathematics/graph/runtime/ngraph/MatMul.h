/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_MATMUL_NGRAPH_H_
#define SHOGUN_MATMUL_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/MatMul.h>
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
				IGNORE_IN_CLASSLIST class MatMulNGraph
				    : public RuntimeNodeTemplate<node::MatMul, ::ngraph::Node>
				{
				public:
					MatMulNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "MatMul";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						const auto input_node =
						    std::static_pointer_cast<node::MatMul>(node);
						if (m_input_nodes.size() != 2)
							error("Expected two input nodes in "
							      "MatMulNGraph.");
						const bool transpose_a = input_node->get_transpose_a();
						const bool transpose_b = input_node->get_transpose_b();

						std::shared_ptr<::ngraph::Node> input1 =
						    m_input_nodes[0];
						std::shared_ptr<::ngraph::Node> input2 =
						    m_input_nodes[1];
						return std::make_shared<::ngraph::op::MatMul>(
						    input1, input2, transpose_a, transpose_b);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

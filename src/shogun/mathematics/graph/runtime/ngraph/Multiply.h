/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_MULTIPLY_NGRAPH_H_
#define SHOGUN_MULTIPLY_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Multiply.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

#include <ngraph/op/multiply.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class MultiplyNGraph

				    : public RuntimeNodeTemplate<node::Multiply, ::ngraph::Node>
				{
				public:
					MultiplyNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Multiply";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						if (m_input_nodes.size() != 2)
							error(
							    "Expected two input nodes in MultiplyNGraph.");
						auto multiply_node =
						    std::static_pointer_cast<node::Multiply>(node);
						if (multiply_node->get_binary_tensor_compatibility() ==
						    node::BinaryNode::BinaryShapeCompatibity::
						        ArrayArray)
						{
							return std::make_shared<::ngraph::op::Multiply>(
							    m_input_nodes[0], m_input_nodes[1]);
						}
						else if (
						    multiply_node->get_binary_tensor_compatibility() ==
						    node::BinaryNode::BinaryShapeCompatibity::
						        ArrayScalar)
						{
							return std::make_shared<::ngraph::op::Multiply>(
							    m_input_nodes[0], m_input_nodes[1],
							    ::ngraph::op::AutoBroadcastSpec(
							        ::ngraph::op::AutoBroadcastType::NUMPY));
						}
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

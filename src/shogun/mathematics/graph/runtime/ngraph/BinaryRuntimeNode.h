/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_BINARY_RUNTIME_NODE_NGRAPH_H_
#define SHOGUN_BINARY_RUNTIME_NODE_NGRAPH_H_

#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

#include <ngraph/op/add.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				template <typename NodeType, typename OperatorType>
				IGNORE_IN_CLASSLIST class BinaryRuntimeNodeNGraph

				    : public RuntimeNodeTemplate<NodeType, ::ngraph::Node>
				{
				public:
					BinaryRuntimeNodeNGraph()
					    : RuntimeNodeTemplate<NodeType, ::ngraph::Node>(){}

		          	[[nodiscard]] std::shared_ptr<::ngraph::Node> build_implementation(
		                  const std::shared_ptr<node::Node>& node) const final
					{
						if (this->m_input_nodes.size() != 2)
							error("Expected two input nodes in "
							      "BinaryRuntimeNodeNGraph.");

						std::shared_ptr<::ngraph::Node> result = nullptr;

						auto binary_node =
						    std::static_pointer_cast<NodeType>(node);
						if (binary_node->get_binary_tensor_compatibility() ==
						    node::BinaryNode::BinaryShapeCompatibity::
						        ArrayArray)
						{
							result = std::make_shared<OperatorType>(
							    this->m_input_nodes[0], this->m_input_nodes[1]);
						}
						else if (
						    binary_node->get_binary_tensor_compatibility() ==
						    node::BinaryNode::BinaryShapeCompatibity::
						        ArrayScalar)
						{
							result = std::make_shared<OperatorType>(
							    this->m_input_nodes[0], this->m_input_nodes[1],
							    ::ngraph::op::AutoBroadcastSpec(
							        ::ngraph::op::AutoBroadcastType::NUMPY));
						}
						return result;
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

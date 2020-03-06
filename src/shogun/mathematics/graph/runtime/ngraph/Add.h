/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ADD_NGRAPH_H_
#define SHOGUN_ADD_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Add.h>
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
				IGNORE_IN_CLASSLIST class AddNGraph

				    : public RuntimeNodeTemplate<node::Add, ::ngraph::Node>
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
						if (m_input_nodes.size() != 2)
							error("Expected two input nodes in AddNGraph.");

						auto add_node = std::static_pointer_cast<node::Add>(node);
						if (add_node->get_binary_tensor_compatibility() == node::BinaryNode::BinaryShapeCompatibity::ArrayArray)
						{
							return std::make_shared<::ngraph::op::Add>(
							    m_input_nodes[0], m_input_nodes[1]);
						}
						else if (add_node->get_binary_tensor_compatibility() == node::BinaryNode::BinaryShapeCompatibity::ArrayScalar)
						{
							return std::make_shared<::ngraph::op::Add>(
								    m_input_nodes[0], m_input_nodes[1],
									::ngraph::op::AutoBroadcastSpec(::ngraph::op::AutoBroadcastType::NUMPY));
						}
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

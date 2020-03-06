/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODES_BINARY_LOGICAL_NODE_H_
#define SHOGUN_NODES_BINARY_LOGICAL_NODE_H_

#include <shogun/mathematics/graph/nodes/BinaryNode.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class LogicalBinaryNode : public BaseBinaryNode
			{
			public:
				LogicalBinaryNode(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				    : BaseBinaryNode(
				          node1, node2, BaseBinaryNode::check_shape_compatible(node1, node2),
				          check_type_compatible(node1, node2))
				{
				}

			private:
				element_type check_type_compatible(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				{
					const auto& node1_types = node1->get_types();
					const auto& node2_types = node2->get_types();

					if (node1_types.size() > 1)
						error(
						    "Expected first node to have only one output "
						    "tensor, but got {}",
						    node1_types.size());

					if (node2_types.size() > 1)
						error(
						    "Expected second node to have only one output "
						    "tensor, but got {}",
						    node2_types.size());

					if (node1_types[0] != element_type::BOOLEAN)
						error("Expected type of first node to be bool");
					if (node2_types[0] != element_type::BOOLEAN)
						error("Expected type of second node to be bool");

					return element_type::BOOLEAN;
				}
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
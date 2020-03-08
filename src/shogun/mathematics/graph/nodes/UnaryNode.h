/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODES_UNARY_NODE_H_
#define SHOGUN_NODES_UNARY_NODE_H_

#include <shogun/mathematics/graph/nodes/Node.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class UnaryNode : public Node
			{
			public:
				UnaryNode(const std::shared_ptr<Node>& node)
				    : Node({node}, check_shape(node), check_type(node))
				{
				}

				bool requires_column_major_conversion() const final
				{
					return false;
				}

			protected:
				element_type check_type(const std::shared_ptr<Node>& node)
				{
					const auto& node_types = node->get_types();

					if (node_types.size() > 1)
						error(
						    "Expected node to have only one output "
						    "tensor, but got {}",
						    node_types.size());

					return node_types[0]->type();
				}

				Shape check_shape(const std::shared_ptr<Node>& node)
				{
					const auto& node_shapes = node->get_shapes();

					if (node_shapes.size() > 1)
						error(
						    "Expected node to have only one output "
						    "tensor, but got {}",
						    node_shapes.size());

					return node_shapes[0];
				}
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
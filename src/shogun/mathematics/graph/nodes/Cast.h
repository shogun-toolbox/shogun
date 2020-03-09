/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODE_CAST_T
#define SHOGUN_NODE_CAST_T

#include <shogun/mathematics/graph/nodes/Node.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class Cast : public Node
			{
			public:
				Cast(const std::shared_ptr<Node>& node, element_type type)
				    : Node({node}, check_shape(node), check_type(node, type))
				{
				}

				std::string to_string() const final
				{
					return fmt::format(
					    "Cast(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const final
				{
					return "Cast";
				}

				bool requires_column_major_conversion() const final
				{
					return false;
				}

			private:
				element_type check_type(
				    const std::shared_ptr<Node>& node, const element_type& type)
				{
					const auto& node_types = node->get_types();

					if (node_types.size() > 1)
						error(
						    "Expected node to have only one output "
						    "tensor, but got {}",
						    node_types.size());

					auto cast_type = number_type(type);
					if (!cast_type->compatible(*node_types[0]))
						error(
						    "There's no safe way to cast {} to {}",
						    node_types[0]->to_string(), cast_type->to_string());

					return type;
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
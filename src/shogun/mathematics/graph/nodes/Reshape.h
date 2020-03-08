/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODES_RESHAPE_NODE_H_
#define SHOGUN_NODES_RESHAPE_NODE_H_

#include "Input.h"
#include <shogun/mathematics/graph/nodes/Node.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class Reshape : public Node
			{
			public:
				Reshape(
				    const std::shared_ptr<Node>& A, const Shape& output_shape)
				    : Node(
				          {A}, check_shape_compatible(A, output_shape),
				          check_type_compatible(A))
				{
				}

				std::string to_string() const final
				{
					return fmt::format(
					    "Reshape(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const final
				{
					return "Reshape";
				}

				bool requires_column_major_conversion() const final
				{
					return false;
				}

			protected:
				element_type
				check_type_compatible(const std::shared_ptr<Node>& A)
				{
					const auto& node1_types = A->get_types();
					if (node1_types.size() > 1)
						error(
						    "Expected first node to have only one output "
						    "tensor, but got {}",
						    node1_types.size());
					return node1_types[0]->type();
				}

				Shape check_shape_compatible(
				    const std::shared_ptr<Node>& A, const Shape& output_shape)
				{
					const auto& node1_shapes = A->get_shapes();

					if (node1_shapes.size() > 1)
						error(
						    "Expected first node to have only one output "
						    "tensor, but got {}",
						    node1_shapes.size());
					const auto& shape = node1_shapes[0];

					if (shape.is_static())
					{
						auto original_size = std::accumulate(
						    shape.begin(), shape.end(), 1, std::multiplies{});
						auto reshape_size = std::accumulate(
						    output_shape.begin(), output_shape.end(), 1,
						    std::multiplies{});
						if (original_size != reshape_size)
						{
							error(
							    "Reshape operation with shapes {} and {} "
							    "modifies "
							    "total size of tensor from {} to {}!",
							    shape.to_string(), output_shape.to_string(),
							    original_size, reshape_size);
						}
					}

					return output_shape;
				}
			};

		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
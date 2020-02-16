/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSORBINARYNODE_H_
#define SHOGUNTENSORBINARYNODE_H_

#include <shogun/mathematics/graph/nodes/Node.h>
#include <shogun/util/enumerate.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class BinaryNode : public Node
			{
			public:
				BinaryNode(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				    : Node(
				          {node1, node2}, check_shape_compatible(node1, node2),
				          check_type_compatible(node1, node2))
				{
				}

			protected:
				element_type check_type_compatible(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				{
					auto node1_types = node1->get_types();
					auto node2_types = node2->get_types();

					if (node1_types.size() > 1)
						error(
						    "Expected first node to have only one output "
						    "tensor, but "
						    "got {}",
						    node1_types.size());

					if (node2_types.size() > 1)
						error(
						    "Expected second node to have only one output "
						    "tensor, but "
						    "got {}",
						    node2_types.size());

					if (node1_types[0] != node2_types[0])
						error("Expected types to be the same");

					return node1_types[0];
				}

				Shape check_shape_compatible(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				{
					auto node1_shapes = node1->get_shapes();
					auto node2_shapes = node2->get_shapes();

					if (node1_shapes.size() > 1)
						error(
						    "Expected first node to have only one output "
						    "tensor, but "
						    "got {}",
						    node1_shapes.size());

					if (node2_shapes.size() > 1)
						error(
						    "Expected second node to have only one output "
						    "tensor, but "
						    "got {}",
						    node2_shapes.size());

					if (node1_shapes[0].size() != node2_shapes[0].size())
					{
						error(
						    "Number of dimension mismatch between {} and {}.",
						    node1, node2);
					}

					std::vector<Shape::shape_type> output_shape_vector;

					for (const auto& [idx, shape1, shape2] :
					     enumerate(node1_shapes[0], node2_shapes[0]))
					{
						if (shape1 == shape2)
						{
							output_shape_vector.push_back(shape1);
						}
						else if (
						    shape1 == Shape::Dynamic &&
						    shape2 == Shape::Dynamic)
						{
							output_shape_vector.push_back(Shape::Dynamic);
						}
						else if (
						    shape1 != Shape::Dynamic &&
						    shape2 != Shape::Dynamic && shape1 != shape2)
						{
							// this is a mismatch, it can't possible go well at
							// runtime
							error(
							    "Shape mismatch in dimension {} when comparing "
							    "{} and "
							    "{}",
							    idx, shape1, shape2);
						}
						else if (shape1 == Shape::Dynamic)
						{
							// shape2 is more restrictive so pick that one
							output_shape_vector.push_back(shape2);
						}
						else if (shape2 == Shape::Dynamic)
						{
							// shape1 is more restrictive so pick that one
							output_shape_vector.push_back(shape1);
						}
						else
						{
							error("Unexpected path: contact a dev or raise an "
							      "issue!");
						}
					}

					return Shape{output_shape_vector};
				}
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
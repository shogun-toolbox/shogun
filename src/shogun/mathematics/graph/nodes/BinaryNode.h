/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODES_BINARY_NODE_H_
#define SHOGUN_NODES_BINARY_NODE_H_

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
				enum class BinaryShapeCompatibity
				{
					ArrayArray = 0,
					ArrayScalar = 1,
					BroadcastAlongAxis = 2
				};

				BinaryNode(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				    : Node(
				          {node1, node2}, check_shape_compatible(node1, node2),
				          check_type_compatible(node1, node2))
				{
				}

				BinaryShapeCompatibity get_binary_tensor_compatibility() const
				{
					return m_shape_compatibility;
				}

				bool requires_column_major_conversion() const final
				{
					return false;
				}

			protected:
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

					if (node1_types[0] != node2_types[0])
						error("Expected types to be the same");

					return node1_types[0];
				}

				Shape check_shape_compatible(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				{
					// by default assume that this is going to be a binary operation 
					// of two nodes with the same number of elements
					m_shape_compatibility = BinaryShapeCompatibity::ArrayArray;

					const auto& node1_shapes = node1->get_shapes();
					const auto& node2_shapes = node2->get_shapes();

					if (node1_shapes.size() > 1)
						error(
						    "Expected first node to have only one output "
						    "tensor, but got {}",
						    node1_shapes.size());

					if (node2_shapes.size() > 1)
						error(
						    "Expected second node to have only one output "
						    "tensor, but got {}",
						    node2_shapes.size());

					const auto& node1_shape = node1_shapes[0];
					const auto& node2_shape = node2_shapes[0];

					if (node1_shape.is_scalar() || node2_shape.is_scalar())
					{
						return scalar_binary_op(node1_shape, node2_shape);
					}
					else if (node1_shape.size() != node2_shape.size())
					{
						error(
						    "BinaryNode error. "
						    "Number of dimension mismatch between {} and {}.",
						    node1->to_string(), node2->to_string());
					}
					return same_shape_binary_op(node1_shape, node2_shape);
				}

			private:
				Shape scalar_binary_op(const Shape& node1_shape, const Shape& node2_shape)
				{
					// xor shapes -> one is a scalar other is an array
					if (!node1_shape.is_scalar() != !node2_shape.is_scalar())
						m_shape_compatibility = BinaryShapeCompatibity::ArrayScalar;
					if (node1_shape.is_scalar())
						return node2_shape;
					if (node2_shape.is_scalar())
						return node1_shape;
				}

				Shape same_shape_binary_op(const Shape& node1_shape, const Shape& node2_shape)
				{
					std::vector<Shape::shape_type> output_shape_vector;

					for (const auto& [idx, shape1, shape2] :
					     enumerate(node1_shape, node2_shape))
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
							    "{} and {}",
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
							error("BinaryNode: Unexpected path, contact a dev or raise an "
							      "issue!");
						}
					}

					return Shape{output_shape_vector};
				}

				BinaryShapeCompatibity m_shape_compatibility;
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
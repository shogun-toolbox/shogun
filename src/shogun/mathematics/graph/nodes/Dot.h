/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODES_DOT_NODE_H_
#define SHOGUN_NODES_DOT_NODE_H_

#include <shogun/mathematics/graph/nodes/Node.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{

			IGNORE_IN_CLASSLIST class Dot : public Node
			{
				friend class MatMul;

			public:
				Dot(const std::shared_ptr<Node>& A,
				    const std::shared_ptr<Node>& B)
				    : Node(
				          {A, B}, check_shape_compatible_helper(A, B),
				          check_type_compatible(A, B))
				{
				}

				const auto& get_reduction_axis_a()
				{
					return m_reduction_axis_a;
				}

				const auto& get_reduction_axis_b()
				{
					return m_reduction_axis_b;
				}

				std::string to_string() const final
				{
					return fmt::format(
					    "Dot(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const final
				{
					return "Dot";
				}

				bool requires_column_major_conversion() const final
				{
					return true;
				}

			protected:
				element_type check_type_compatible(
				    const std::shared_ptr<Node>& A,
				    const std::shared_ptr<Node>& B);

				Shape check_shape_compatible_helper(
				    const std::shared_ptr<Node>& A,
				    const std::shared_ptr<Node>& B)
				{
					auto [result, reduction_axis_a, reduction_axis_b] =
					    check_shape_compatible_(A, B);
					m_reduction_axis_a = reduction_axis_a;
					m_reduction_axis_b = reduction_axis_b;

					return result;
				}

				std::tuple<Shape, size_t, size_t> check_shape_compatible_(
				    const std::shared_ptr<Node>& A,
				    const std::shared_ptr<Node>& B)
				{
					const auto& node_a_shapes = A->get_shapes();
					const auto& node_b_shapes = B->get_shapes();

					if (node_a_shapes.size() > 1)
						error(
						    "Expected first node to have only one output "
						    "tensor, but got {}",
						    node_a_shapes.size());

					if (node_b_shapes.size() > 1)
						error(
						    "Expected second node to have only one output "
						    "tensor, but got {}",
						    node_b_shapes.size());

					const auto& shape_a = node_a_shapes[0];
					const auto& shape_b = node_b_shapes[0];

					return check_shape_compatible(shape_a, shape_b);
				}

				static std::tuple<Shape, size_t, size_t> check_shape_compatible(
				    const Shape& shape_a, const Shape& shape_b);
			private:
				size_t m_reduction_axis_a;
				size_t m_reduction_axis_b;
			};

		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
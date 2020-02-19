/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODES_MATMUL_NODE_H_
#define SHOGUN_NODES_MATMUL_NODE_H_

#include <shogun/mathematics/graph/nodes/Node.h>
#include <shogun/util/enumerate.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class Dot : public Node
			{
			public:
				Dot(const std::shared_ptr<Node>& A,
				    const std::shared_ptr<Node>& B)
				    : Node(
				          {A, B}, check_shape_compatible(A, B),
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

			protected:
				element_type check_type_compatible(
				    const std::shared_ptr<Node>& A,
				    const std::shared_ptr<Node>& B)
				{
					const auto& node1_types = A->get_types();
					const auto& node2_types = B->get_types();

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

					m_reduction_axis_a = shape_a.size() - 1;
					m_reduction_axis_b =
					    shape_b.size() <= 1 ? 0 : shape_b.size() - 2;

					if (!shape_a.partial_compare(
					        m_reduction_axis_a, shape_b[m_reduction_axis_b]))
						error(
						    "shapes {} and {} not aligned: {} (dim {}) != {} "
						    "(dim {})",
						    shape_a.to_string(), shape_b.to_string(),
						    shape_a[m_reduction_axis_a], m_reduction_axis_a,
						    shape_b[m_reduction_axis_b], m_reduction_axis_b);

					std::vector<Shape::shape_type> output_shape_vector;

					for (const auto& [idx, el] : enumerate(shape_a))
					{
						if (idx != m_reduction_axis_a)
							output_shape_vector.push_back(el);
					}

					for (const auto& [idx, el] : enumerate(shape_b))
					{
						if (idx != m_reduction_axis_b)
							output_shape_vector.push_back(el);
					}

					return Shape{output_shape_vector};
				}

			private:
				index_t m_reduction_axis_a;
				index_t m_reduction_axis_b;
			};

		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
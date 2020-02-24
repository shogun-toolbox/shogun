/*
 * This software is distributed under BSD 3-clause license (see LICENSE
 file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODES_MATMUL_NODE_H_
#define SHOGUN_NODES_MATMUL_NODE_H_

#include <shogun/mathematics/graph/nodes/Node.h>
#include <shogun/mathematics/graph/nodes/Dot.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{

			// The behavior depends on the arguments in the following way.
			// -    If both arguments are 2-D they are multiplied like
			// 		conventional matrices.
			// -    If either argument is N-D, N > 2, it is treated as a stack
			// 		of matrices residing in the last two indexes and broadcast
			// 		accordingly.
			// -    If the first argument is 1-D, it is promoted to a matrix by
			// 		prepending a 1 to its dimensions. After matrix multiplication 
			// 		the	prepended 1 is removed.
			// -    If the second argument is 1-D, it is promoted to a matrix by
			// 		appending a 1 to its dimensions. After matrix multiplication
			//      the	appended 1 is removed. 
			// Matmul differs from dot in two important ways:
			//  -	Multiplication by scalars is not allowed, use Dot instead.
			//  -	Stacks of matrices are broadcast together as if the matrices
			//  	were elements, respecting the signature (n,k),(k,m)->(n,m):
			IGNORE_IN_CLASSLIST class MatMul : public Node
			{
			public:
				MatMul(
				    const std::shared_ptr<Node>& A,
				    const std::shared_ptr<Node>& B, bool transpose_a,
				    bool transpose_b)
				    : Node(
				          {A, B}, check_shape_compatible(A, B),
				          check_type_compatible(A, B)),
				      m_transpose_a(transpose_a), m_transpose_b(transpose_b)
				{
				}

				MatMul(
				    const std::shared_ptr<Node>& A,
				    const std::shared_ptr<Node>& B): MatMul(A, B, false, false)
				{
				}

				bool get_transpose_a()
				{
					return m_transpose_a;
				}

				bool get_transpose_b()
				{
					return m_transpose_b;
				}

				std::string to_string() const final
				{
					return fmt::format(
					    "MatMul(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const final
				{
					return "MatMul";
				}

				bool requires_column_major_conversion() const final
				{
					return true;
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

					if (node_a_shapes[0].is_scalar())
						error(
						    "Node A is a scalar: {}",
						    node_a_shapes[0].to_string());

					if (node_b_shapes[0].is_scalar())
						error(
						    "Node A is a scalar: {}",
						    node_b_shapes[0].to_string());

					if (node_a_shapes[0].size() != node_b_shapes[0].size())
					{
						error(
						    "Number of dimension mismatch between {} and {}.",
						    A, B);
					}

					auto shape_a = node_a_shapes[0];
					auto shape_b = node_b_shapes[0];

					if (shape_a.size() > 2)
						error(
						    "Currently MatMul cannot handle tensors with more "
						    "than two dimensions. "
						    "Node A output tensor has {} dimensions.",
						    shape_a.size());

					if (shape_b.size() > 2)
						error(
						    "Currently MatMul cannot handle tensors with more "
						    "than two dimensions. "
						    "Node B output tensor has {} dimensions.",
						    shape_b.size());

					if (m_transpose_a)
						shape_a = shape_a.switch_major();
					if (m_transpose_b)
						shape_b = shape_b.switch_major();

					auto [result, reduction_axis_a, reduction_axis_b] = Dot::check_shape_compatible(shape_a, shape_b);
					m_reduction_axis_a = reduction_axis_a;
					m_reduction_axis_b = reduction_axis_b;
					return result;

					// std::vector<Shape::shape_type> output_shape_vector;
					// if (m_transpose_a && !m_transpose_b)
					// {
					// 	if (shape_a[1] == Shape::Dynamic &&
					// 	    shape_b[1] == Shape::Dynamic)
					// 		output_shape_vector[0] = Shape::Dynamic;
					// 	else if (
					// 	    shape_a[1] == Shape::Dynamic &&
					// 	    shape_b[1] != Shape::Dynamic)
					// 		output_shape_vector[0] = shape_b[1];
					// 	else if (
					// 	    shape_a[1] != Shape::Dynamic &&
					// 	    shape_b[1] == Shape::Dynamic)
					// 		output_shape_vector[0] = shape_a[1];
					// 	else if (shape_a[1] != shape_b[1])
					// 		error("MatMul A transpose tensor dimension mismatch. "
					// 			  "{} vs {}", shape_a, shape_b); 						
					// 	else
					// 		error("Undefined state...");
					// }
					// else if (m_transpose_b && !m_transpose_a)
					// {
					// 	if (shape_a[0] == Shape::Dynamic &&
					// 	    shape_b[0] == Shape::Dynamic)
					// 		output_shape_vector[1] = Shape::Dynamic;
					// 	else if (
					// 	    shape_a[0] == Shape::Dynamic &&
					// 	    shape_b[0] != Shape::Dynamic)
					// 		output_shape_vector[1] = shape_b[0];
					// 	else if (
					// 	    shape_a[0] != Shape::Dynamic &&
					// 	    shape_b[0] == Shape::Dynamic)
					// 		output_shape_vector[1] = shape_a[0];
					// 	else if (shape_a[1] != shape_b[1])
					// 		error(
					// 		    "MatMul A transpose tensor dimension "
					// 		    "mismatch. {} vs {}",
					// 		    shape_a, shape_b);
					// 	else
					// 		error("Undefined state...");
					// }
					// else if (m_transpose_a && m_transpose_b)
					// {
					// 	if (shape_a[1] != shape_b[0])
					// 		error(
					// 		    "MatMul B transpose tensor dimension "
					// 		    "mismatch. {} vs {}",
					// 		    shape_a, shape_b);
					// 	output_shape_vector[0] = shape_a[1];
					// 	output_shape_vector[1] = shape_a[0];
					// }
					// else if (shape_a[0] == Shape::Dynamic &&
					// 	    shape_b[1] != Shape::Dynamic)
					// {
					// }
					// else
					// {
					// 	error("TODO");
					// }

					// return Shape{output_shape_vector};
				}

			private:
				const bool m_transpose_a;
				const bool m_transpose_b;
				size_t m_reduction_axis_a;
				size_t m_reduction_axis_b;
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
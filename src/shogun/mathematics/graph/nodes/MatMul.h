// /*
//  * This software is distributed under BSD 3-clause license (see LICENSE
//  file).
//  *
//  * Authors: Gil Hoben
//  */

// #ifndef SHOGUN_NODES_MATMUL_NODE_H_
// #define SHOGUN_NODES_MATMUL_NODE_H_

// #include <shogun/mathematics/graph/nodes/Node.h>
// #include <shogun/util/enumerate.h>

// #define IGNORE_IN_CLASSLIST

// namespace shogun
// {
// 	namespace graph
// 	{
// 		namespace node
// 		{
// 			IGNORE_IN_CLASSLIST class MatMul : public Node
// 			{
// 			public:
// 				MatMul(
// 				    const std::shared_ptr<Node>& A,
// 				    const std::shared_ptr<Node>& B,
// 				    bool transpose_a,
// 				    bool transpose_b)
// 				    : Node(
// 				          {A, B}, check_shape_compatible(A, B),
// 				          check_type_compatible(A, B))
// 				    , m_transpose_a(transpose_a)
// 				    , m_transpose_b(transpose_b)
// 				{
// 				}

// 			bool get_transpose_a()
// 			{
// 				return m_transpose_a;
// 			}

// 			bool get_transpose_b()
// 			{
// 				return m_transpose_b;
// 			}

// 			protected:
// 				element_type check_type_compatible(
// 				    const std::shared_ptr<Node>& A,
// 				    const std::shared_ptr<Node>& B)
// 				{
// 					const auto& node1_types = A->get_types();
// 					const auto& node2_types = B->get_types();

// 					if (node1_types.size() > 1)
// 						error(
// 						    "Expected first node to have only one output "
// 						    "tensor, but got {}",
// 						    node1_types.size());

// 					if (node2_types.size() > 1)
// 						error(
// 						    "Expected second node to have only one output "
// 						    "tensor, but got {}",
// 						    node2_types.size());

// 					if (node1_types[0] != node2_types[0])
// 						error("Expected types to be the same");

// 					return node1_types[0];
// 				}

// 				Shape check_shape_compatible(
// 				    const std::shared_ptr<Node>& A,
// 				    const std::shared_ptr<Node>& B)
// 				{
// 					const auto& node_a_shapes = A->get_shapes();
// 					const auto& node_b_shapes = B->get_shapes();

// 					if (node_a_shapes.size() > 1)
// 						error(
// 						    "Expected first node to have only one output "
// 						    "tensor, but got {}",
// 						    node1_shapes.size());

// 					if (node_b_shapes.size() > 1)
// 						error(
// 						    "Expected second node to have only one output "
// 						    "tensor, but got {}",
// 						    node_b_shapes.size());

// 					if (node_a_shapes[0].size() != node_b_shapes[0].size())
// 					{
// 						error(
// 						    "Number of dimension mismatch between {} and {}.",
// 						    node1, node2);
// 					}

// 					const auto& shape_a = node_a_shapes[0];
// 					const auto& shape_b = node_b_shapes[0];

// 					if (shape_a.size() <= 2)
// 						error("Currently MatMul cannot handle tensors with more
// than two dimensions. " 							"Node A output tensor has {}
// dimensions.", shape_a.size());

// 					if (shape_b.size() <= 2)
// 						error("Currently MatMul cannot handle tensors with more
// than two dimensions. " 							"Node B output tensor has {}
// dimensions.", shape_b.size());

// 					std::vector<Shape::shape_type> output_shape_vector;
// 					if (shape_a.size() == 1 && shape_b.size() == 1)
// 					{
// 					}
// 					else if ()

// 					if (m_transpose_a && !m_transpose_b)
// 					{
// 						if (shape_a[1] == Shape::Dynamic && shape_b[1] ==
// Shape::Dynamic) 							output_shape_vector[0] = Shape::Dynamic; 						else
// if (shape_a[1]
// == Shape::Dynamic && shape_b[1] != Shape::Dynamic)
// output_shape_vector[0] = shape_b[1]; 						else if
// (shape_a[1] != Shape::Dynamic && shape_b[1] == Shape::Dynamic)
// output_shape_vector[0] = shape_a[1]; 						else if
// (shape_a[1]
// !=
// shape_b[1]) 							error("MatMul A transpose tensor dimension mismatch. {}
// vs
// {}", shape_a, shape_b); 						else
// error("Undefined state...");
// 					}
// 					else if (m_transpose_b && !m_transpose_a)
// 					{
// 						if (shape_a[0] == Shape::Dynamic && shape_b[0] ==
// Shape::Dynamic) 							output_shape_vector[1] = Shape::Dynamic; 						else
// if (shape_a[0]
// == Shape::Dynamic && shape_b[0] != Shape::Dynamic)
// output_shape_vector[1] = shape_b[0]; 						else if
// (shape_a[0] != Shape::Dynamic && shape_b[0] == Shape::Dynamic)
// output_shape_vector[1] = shape_a[0]; 						else if
// (shape_a[1]
// !=
// shape_b[1]) 							error("MatMul A transpose tensor dimension mismatch. {}
// vs
// {}", shape_a, shape_b); 						else
// error("Undefined state...");
// 					}
// 					else if (m_transpose_a && m_transpose_b)
// 					{
// 						if (shape_a[1] != shape_b[0])
// 							error("MatMul B transpose tensor dimension mismatch. {}
// vs
// {}", shape_a, shape_b) 						output_shape_vector[0] =
// shape_a[1]; 						output_shape_vector[1] = shape_a[0];
// 					}
// 					else
// 					{
// 						if (shape_a[0] == Shape::Dynamic && shape_b[1] !=
// Shape::Dynamic)

// 					}

// 					return Shape{output_shape_vector};
// 				}

// 			private:
// 				const bool m_transpose_a;
// 				const bool m_transpose_b;
// 			};
// 		} // namespace node
// 	}     // namespace graph
// } // namespace shogun

// #endif
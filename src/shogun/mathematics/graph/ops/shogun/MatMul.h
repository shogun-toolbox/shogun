/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_MATMUL_SHOGUN_H_
#define SHOGUN_MATMUL_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/MatMul.h>
#include <shogun/mathematics/graph/ops/abstract/Operator.h>
#include <shogun/mathematics/graph/ops/shogun/Dot.h>

#include <shogun/mathematics/eigen3.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class MatMulShogun : public Operator
			{
			public:
				MatMulShogun(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "MatMul";
				}

				void call(const std::vector<std::shared_ptr<
				              detail::shogun::OutputNode>>& input_nodes) final
				{
					if (input_nodes.size() != 2)
						error("Binary operation expected two inputs.");
					if (m_outputs.size() != 1)
						error("Binary operation expected one output.");

					const bool transpose_a =
					    std::static_pointer_cast<node::MatMul>(m_node)
					        ->get_transpose_a();
					const bool transpose_b =
					    std::static_pointer_cast<node::MatMul>(m_node)
					        ->get_transpose_b();

					const auto& input1 = input_nodes[0]->get_outputs()[0];
					const auto& input2 = input_nodes[1]->get_outputs()[0];
					const auto& output = m_outputs[0];

					runtime_checks_and_allocation(
					    input1, input2, transpose_a, transpose_b);

					matmul_type_dispatch(
					    input1, input2, output, transpose_a, transpose_b);
				}

			private:
				void runtime_checks_and_allocation(
				    const std::shared_ptr<ShogunStorage>& input1,
				    const std::shared_ptr<ShogunStorage>& input2,
				    const bool transpose_a, const bool transpose_b)
				{
					const auto& shape_a = input1->get_shape();
					const auto& shape_b = input2->get_shape();

					auto reduction_axis_a =
					    std::static_pointer_cast<node::MatMul>(m_node)
					        ->get_reduction_axis_a();
					auto reduction_axis_b =
					    std::static_pointer_cast<node::MatMul>(m_node)
					        ->get_reduction_axis_b();

					if (transpose_a)
						reduction_axis_a = std::abs(
						    static_cast<int64_t>(reduction_axis_a) - 1);

					if (transpose_b)
						reduction_axis_b = std::abs(
						    static_cast<int64_t>(reduction_axis_b) - 1);

					if (shape_a[reduction_axis_a] != shape_b[reduction_axis_b])
					{
						error(
						    "Runtime MatMul shape mismatch. "
						    "shapes {} and {} not aligned: {} (dim {}) != "
						    "{} (dim {})",
						    shape_a.to_string(), shape_b.to_string(),
						    shape_a[reduction_axis_a], reduction_axis_a,
						    shape_b[reduction_axis_b], reduction_axis_b);
					}

					std::vector<Shape::shape_type> output_shape_vector;

					for (const auto& [idx, el] : enumerate(shape_a))
					{
						if (idx != reduction_axis_a)
							output_shape_vector.push_back(el);
					}

					for (const auto& [idx, el] : enumerate(shape_b))
					{
						if (idx != reduction_axis_b)
							output_shape_vector.push_back(el);
					}

					m_outputs[0]->allocate_storage(Shape{output_shape_vector});
				}

				void matmul_type_dispatch(
				    const std::shared_ptr<ShogunStorage>& A,
				    const std::shared_ptr<ShogunStorage>& B,
				    const std::shared_ptr<ShogunStorage>& Out,
				    const bool transpose_a, const bool transpose_b)
				{
					if (!transpose_a && !transpose_b)
						DotShogun::dot_product_type_dispatch(A, B, Out);
					else
					{
#define CALL_KERNEL_IMPLEMENTATION(NUMBER_TYPE)                                \
	case NUMBER_TYPE::type_id:                                                 \
		matmul_dispatch<NUMBER_TYPE::c_type>(                                  \
		    A, B, Out, transpose_a, transpose_b);                              \
		break;

						switch (*A->get_type())
						{
							CALL_KERNEL_IMPLEMENTATION(BooleanType)
							CALL_KERNEL_IMPLEMENTATION(Int8Type)
							CALL_KERNEL_IMPLEMENTATION(Int16Type)
							CALL_KERNEL_IMPLEMENTATION(Int32Type)
							CALL_KERNEL_IMPLEMENTATION(Int64Type)
							CALL_KERNEL_IMPLEMENTATION(UInt8Type)
							CALL_KERNEL_IMPLEMENTATION(UInt16Type)
							CALL_KERNEL_IMPLEMENTATION(UInt32Type)
							CALL_KERNEL_IMPLEMENTATION(UInt64Type)
							CALL_KERNEL_IMPLEMENTATION(Float32Type)
							CALL_KERNEL_IMPLEMENTATION(Float64Type)
						}
#undef CALL_KERNEL_IMPLEMENTATION
					}
				}

				template <typename T>
				void matmul_dispatch(
				    const std::shared_ptr<ShogunStorage>& A,
				    const std::shared_ptr<ShogunStorage>& B,
				    const std::shared_ptr<ShogunStorage>& Out,
				    const bool transpose_a, const bool transpose_b)
				{
					Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
					A_eig(
					    static_cast<T*>(A->data()), A->get_shape()[0],
					    A->get_shape()[1]);
					Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
					B_eig(
					    static_cast<T*>(B->data()), B->get_shape()[0],
					    B->get_shape()[1]);
					Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
					Out_eig(
					    static_cast<T*>(Out->data()), Out->get_shape()[0],
					    Out->get_shape()[1]);
					if (transpose_a)
						Out_eig = A_eig.transpose() * B_eig;
					else if (transpose_b)
						Out_eig = A_eig * B_eig.transpose();
					else if (transpose_a && transpose_b)
						Out_eig = A_eig.transpose() * B_eig.transpose();
					else
						Out_eig = A_eig * B_eig;
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
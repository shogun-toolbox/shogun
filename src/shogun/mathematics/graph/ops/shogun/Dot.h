/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DOT_SHOGUN_H_
#define SHOGUN_DOT_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Dot.h>
#include <shogun/mathematics/graph/ops/abstract/Operator.h>

#include <shogun/mathematics/eigen3.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class DotShogun : public Operator
			{
			public:
				DotShogun(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Dot";
				}

				void call(const std::vector<std::shared_ptr<
				              detail::shogun::OutputNode>>& input_nodes) final
				{
					const auto& input_tensor1 =
					    input_nodes[0]->get_output_tensors()[0];
					const auto& input_tensor2 =
					    input_nodes[1]->get_output_tensors()[0];
					const auto& output_tensor = m_output_tensors[0];

					runtime_checks_and_allocation(
					    std::vector{input_tensor1, input_tensor2});

					dot_product_type_dispatch(
					    input_tensor1, input_tensor2, output_tensor);
				}

			private:
				void runtime_checks_and_allocation(
				    const std::vector<std::shared_ptr<Tensor>>& tensors) final
				{
					if (tensors.size() != 2)
						error("Binary operation expected two inputs.");
					if (m_output_tensors.size() != 1)
						error("Binary operation expected one output.");

					const auto& input_tensor1 = tensors[0];
					const auto& input_tensor2 = tensors[1];
					const auto& shape_a = input_tensor1->get_shape();
					const auto& shape_b = input_tensor2->get_shape();

					const auto& reduction_axis_a =
					    std::static_pointer_cast<node::Dot>(m_node)
					        ->get_reduction_axis_a();
					const auto& reduction_axis_b =
					    std::static_pointer_cast<node::Dot>(m_node)
					        ->get_reduction_axis_b();

					if (shape_a.is_scalar())
					{
						m_output_tensors[0]->allocate_tensor(shape_b);
					}
					else if (shape_b.is_scalar())
					{
						m_output_tensors[0]->allocate_tensor(shape_a);
					}
					else
					{
						if (shape_a[reduction_axis_a] !=
						    shape_b[reduction_axis_b])
						{
							error(
							    "Runtime Dot shape mismatch. "
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

						m_output_tensors[0]->allocate_tensor(
						    Shape{output_shape_vector});
					}
				}

				template <typename T>
				void dot_product_dispatch(
				    const std::shared_ptr<Tensor>& input_tensor1,
				    const std::shared_ptr<Tensor>& input_tensor2,
				    const std::shared_ptr<Tensor>& output_tensor)
				{
					if (input_tensor1->get_shape().is_scalar() &&
					    !input_tensor2->get_shape().is_scalar())
						dot_product_container_scalar_implementation<T>(
						    input_tensor2, input_tensor1, output_tensor);
					else if (
					    !input_tensor1->get_shape().is_scalar() &&
					    input_tensor2->get_shape().is_scalar())
						dot_product_container_scalar_implementation<T>(
						    input_tensor1, input_tensor2, output_tensor);
					else if (
					    input_tensor1->get_shape().size() == 1 &&
					    input_tensor2->get_shape().size() == 1)
						dot_product_vector_vector_implementation<T>(
						    input_tensor1, input_tensor2, output_tensor);
					else if (
					    input_tensor1->get_shape().size() == 1 &&
					    input_tensor2->get_shape().size() == 2)
						dot_product_vector_matrix_implementation<T>(
						    input_tensor1, input_tensor2, output_tensor);
					else if (
					    input_tensor1->get_shape().size() == 2 &&
					    input_tensor2->get_shape().size() == 1)
						dot_product_matrix_vector_implementation<T>(
						    input_tensor1, input_tensor2, output_tensor);
					else if (
					    input_tensor1->get_shape().size() == 2 &&
					    input_tensor2->get_shape().size() == 2)
						dot_product_matrix_matrix_implementation<T>(
						    input_tensor1, input_tensor2, output_tensor);
					else
					{
						// this would require using Eigen::Tensor, like in
						// ngraph
						error(
						    "Dot cannot handle the provided shapes: {} and {}",
						    input_tensor1->get_shape().to_string(),
						    input_tensor2->get_shape().to_string());
					}
				}

				void dot_product_type_dispatch(
				    const std::shared_ptr<Tensor>& A,
				    const std::shared_ptr<Tensor>& B,
				    const std::shared_ptr<Tensor>& Out)
				{
#define CALL_KERNEL_IMPLEMENTATION(SHOGUN_TYPE)                                \
	case SHOGUN_TYPE:                                                          \
		dot_product_dispatch<get_type_from_enum<SHOGUN_TYPE>::type>(           \
		    A, B, Out);                                                        \
		break;

					switch (A->get_type())
					{
						CALL_KERNEL_IMPLEMENTATION(element_type::BOOLEAN)
						CALL_KERNEL_IMPLEMENTATION(element_type::INT8)
						CALL_KERNEL_IMPLEMENTATION(element_type::INT16)
						CALL_KERNEL_IMPLEMENTATION(element_type::INT32)
						CALL_KERNEL_IMPLEMENTATION(element_type::INT64)
						CALL_KERNEL_IMPLEMENTATION(element_type::UINT8)
						CALL_KERNEL_IMPLEMENTATION(element_type::UINT16)
						CALL_KERNEL_IMPLEMENTATION(element_type::UINT32)
						CALL_KERNEL_IMPLEMENTATION(element_type::UINT64)
						CALL_KERNEL_IMPLEMENTATION(element_type::FLOAT32)
						CALL_KERNEL_IMPLEMENTATION(element_type::FLOAT64)
					}
#undef CALL_KERNEL_IMPLEMENTATION
				}

				template <typename T>
				void dot_product_container_scalar_implementation(
				    const std::shared_ptr<Tensor>& A,
				    const std::shared_ptr<Tensor>& B,
				    const std::shared_ptr<Tensor>& Out)
				{
					Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> A_eig(
					    static_cast<T*>(A->data()), A->size());
					Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> Out_eig(
					    static_cast<T*>(Out->data()), Out->size());
					Out_eig = A_eig * *static_cast<T*>(B->data());
				}

				template <typename T>
				void dot_product_vector_vector_implementation(
				    const std::shared_ptr<Tensor>& A,
				    const std::shared_ptr<Tensor>& B,
				    const std::shared_ptr<Tensor>& Out)
				{
					Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> A_eig(
					    static_cast<T*>(A->data()), A->size());
					Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> B_eig(
					    static_cast<T*>(B->data()), B->size());

					static_cast<T*>(Out->data())[0] = A_eig.dot(B_eig);
				}

				template <typename T>
				void dot_product_vector_matrix_implementation(
				    const std::shared_ptr<Tensor>& A,
				    const std::shared_ptr<Tensor>& B,
				    const std::shared_ptr<Tensor>& Out)
				{
					Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> A_eig(
					    static_cast<T*>(A->data()), A->size());
					Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
					B_eig(
					    static_cast<T*>(B->data()), B->get_shape()[0],
					    B->get_shape()[1]);
					Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> Out_eig(
					    static_cast<T*>(Out->data()), Out->size());

					Out_eig = A_eig * B_eig;
				}

				template <typename T>
				void dot_product_matrix_vector_implementation(
				    const std::shared_ptr<Tensor>& A,
				    const std::shared_ptr<Tensor>& B,
				    const std::shared_ptr<Tensor>& Out)
				{
					Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
					A_eig(
					    static_cast<T*>(A->data()), A->get_shape()[0],
					    A->get_shape()[1]);
					Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> B_eig(
					    static_cast<T*>(B->data()), B->size());
					Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> Out_eig(
					    static_cast<T*>(Out->data()), Out->size());

					Out_eig = A_eig * B_eig;
				}

				template <typename T>
				void dot_product_matrix_matrix_implementation(
				    const std::shared_ptr<Tensor>& A,
				    const std::shared_ptr<Tensor>& B,
				    const std::shared_ptr<Tensor>& Out)
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

					Out_eig = A_eig * B_eig;
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
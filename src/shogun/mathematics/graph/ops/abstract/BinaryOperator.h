/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_BINARY_OPERATOR_H_
#define SHOGUN_BINARY_OPERATOR_H_

#include <shogun/mathematics/graph/nodes/BinaryNode.h>
#include <shogun/mathematics/graph/ops/abstract/Operator.h>
#include <shogun/mathematics/graph/runtime/shogun/OutputNode.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			template <typename DerivedOperator>
			class ShogunBinaryOperator : public Operator
			{
			public:
				ShogunBinaryOperator(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				virtual ~ShogunBinaryOperator()
				{
				}

				void call(const std::vector<std::shared_ptr<
				              detail::shogun::OutputNode>>& input_nodes) final
				{
					auto binary_node =
					    std::static_pointer_cast<node::BaseBinaryNode>(m_node);
					const auto shape_compatibility =
					    binary_node->get_binary_tensor_compatibility();

					if (input_nodes.size() != 2)
						error("Binary operation expected two inputs.");

					if (m_outputs.size() != 1)
						error("Binary operation expected one output.");

					const auto& input1 = input_nodes[0]->get_outputs()[0];
					const auto& input2 = input_nodes[1]->get_outputs()[0];
					const auto& output = m_outputs[0];

					runtime_checks_and_allocation(
					    input1, input2, shape_compatibility);
					if (shape_compatibility ==
					    node::BaseBinaryNode::BinaryShapeCompatibity::
					        ArrayArray)
					{
						kernel(
						    input1->data(), input2->data(), output->data(),
						    output->size(), input1->get_type());
					}
					else if (
					    shape_compatibility ==
					    node::BaseBinaryNode::BinaryShapeCompatibity::
					        ArrayScalar)
					{
						const bool scalar_first =
						    input1->get_shape().is_scalar();
						kernel_scalar(
						    input1->data(), input2->data(), output->data(),
						    output->size(), input1->get_type(), scalar_first);
					}
				}

			protected:
				void runtime_checks_and_allocation(
				    const std::shared_ptr<Storage>& input1,
				    const std::shared_ptr<Storage>& input2,
				    const node::BaseBinaryNode::BinaryShapeCompatibity&
				        shape_compatibility)
				{
					allocate_storage(runtime_shape_check(
					    input1, input2, shape_compatibility));
				}

				void allocate_storage(const Shape& shape)
				{
					m_outputs[0]->allocate_storage(shape);
				}

				const Shape& runtime_shape_check(
				    const std::shared_ptr<Storage>& input1,
				    const std::shared_ptr<Storage>& input2,
				    const node::BaseBinaryNode::BinaryShapeCompatibity&
				        shape_compatibility)
				{
					if (shape_compatibility ==
					    node::BaseBinaryNode::BinaryShapeCompatibity::
					        ArrayArray)
					{
						for (auto [idx, shape1, shape2] : enumerate(
						         input1->get_shape(), input2->get_shape()))
						{
							if (shape1 != shape2)
							{
								error(
								    "Runtime shape mismatch in dimension {}. "
								    "Got "
								    "{} and {}.",
								    idx, shape1, shape2);
							}
							if (shape1 == Shape::Dynamic)
							{
								error("Could not infer runtime shape.");
							}
						}
						// shapes have to match exactly so can return either one
						return input1->get_shape();
					}
					else if (
					    shape_compatibility ==
					    node::BaseBinaryNode::BinaryShapeCompatibity::
					        ArrayScalar)
					{
						if (input1->get_shape().is_scalar())
							return input2->get_shape();
						return input1->get_shape();
					}
					else
					{
						error("NotImplemented");
					}
					return input1->get_shape();
				}

				void kernel(
				    void* input1, void* input2, void* output, const size_t size,
				    const node::Node::type_info& type)
				{

#define CALL_KERNEL_IMPLEMENTATION(NUMBER_TYPE)                                \
	static_cast<DerivedOperator*>(this)                                        \
	    ->template kernel_implementation<NUMBER_TYPE::c_type>(                 \
	        input1, input2, output, size);                                     \
	break;

					switch (*type)
					{
					case element_type::BOOLEAN:
						CALL_KERNEL_IMPLEMENTATION(BooleanType)
					case element_type::INT8:
						CALL_KERNEL_IMPLEMENTATION(Int8Type)
					case element_type::INT16:
						CALL_KERNEL_IMPLEMENTATION(Int16Type)
					case element_type::INT32:
						CALL_KERNEL_IMPLEMENTATION(Int32Type)
					case element_type::INT64:
						CALL_KERNEL_IMPLEMENTATION(Int64Type)
					case element_type::UINT8:
						CALL_KERNEL_IMPLEMENTATION(UInt8Type)
					case element_type::UINT16:
						CALL_KERNEL_IMPLEMENTATION(UInt16Type)
					case element_type::UINT32:
						CALL_KERNEL_IMPLEMENTATION(UInt32Type)
					case element_type::UINT64:
						CALL_KERNEL_IMPLEMENTATION(UInt64Type)
					case element_type::FLOAT32:
						CALL_KERNEL_IMPLEMENTATION(Float32Type)
					case element_type::FLOAT64:
						CALL_KERNEL_IMPLEMENTATION(Float64Type)
					}
#undef CALL_KERNEL_IMPLEMENTATION
				}

				void kernel_scalar(
				    void* input1, void* input2, void* output, const size_t size,
				    const node::Node::type_info& type, const bool scalar_first)
				{

#define CALL_KERNEL_IMPLEMENTATION(SHOGUN_TYPE)                                \
	static_cast<DerivedOperator*>(this)                                        \
	    ->template kernel_scalar_implementation<SHOGUN_TYPE::c_type>(          \
	        input1, input2, output, size, scalar_first);                       \
	break;

					switch (*type)
					{
					case element_type::BOOLEAN:
						CALL_KERNEL_IMPLEMENTATION(BooleanType)
					case element_type::INT8:
						CALL_KERNEL_IMPLEMENTATION(Int8Type)
					case element_type::INT16:
						CALL_KERNEL_IMPLEMENTATION(Int16Type)
					case element_type::INT32:
						CALL_KERNEL_IMPLEMENTATION(Int32Type)
					case element_type::INT64:
						CALL_KERNEL_IMPLEMENTATION(Int64Type)
					case element_type::UINT8:
						CALL_KERNEL_IMPLEMENTATION(UInt8Type)
					case element_type::UINT16:
						CALL_KERNEL_IMPLEMENTATION(UInt16Type)
					case element_type::UINT32:
						CALL_KERNEL_IMPLEMENTATION(UInt32Type)
					case element_type::UINT64:
						CALL_KERNEL_IMPLEMENTATION(UInt64Type)
					case element_type::FLOAT32:
						CALL_KERNEL_IMPLEMENTATION(Float32Type)
					case element_type::FLOAT64:
						CALL_KERNEL_IMPLEMENTATION(Float64Type)
					}
#undef CALL_KERNEL_IMPLEMENTATION
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
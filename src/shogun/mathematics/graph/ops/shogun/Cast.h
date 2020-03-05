/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_CAST_SHOGUN_H_
#define SHOGUN_CAST_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Cast.h>
#include <shogun/mathematics/graph/ops/abstract/Operator.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class CastShogun : public Operator
			{
			public:
				CastShogun(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Cast";
				}

				void call(const std::vector<std::shared_ptr<
				              detail::shogun::OutputNode>>& input_nodes) final
				{
					const auto& input_tensor =
					    input_nodes[0]->get_output_tensors()[0];

					const auto& output_tensor = m_output_tensors[0];

					runtime_checks_and_allocation(input_tensor);

					cast_implementation_type_dispatch(
					    input_tensor, output_tensor);
				}

			protected:
				void runtime_checks_and_allocation(
				    const std::shared_ptr<Tensor>& tensor)
				{
					const auto& shape = tensor->get_shape();
					m_output_tensors[0]->allocate_tensor(shape);
				}

				void cast_implementation_type_dispatch(
				    const std::shared_ptr<Tensor>& input,
				    const std::shared_ptr<Tensor>& output)
				{
#define CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
    INPUT_SHOGUN_TYPE, OUTPUT_SHOGUN_TYPE)                                     \
	case OUTPUT_SHOGUN_TYPE:                                                   \
		kernel_implementation<                                                 \
		    get_type_from_enum<INPUT_SHOGUN_TYPE>::type,                       \
		    get_type_from_enum<OUTPUT_SHOGUN_TYPE>::type>(                     \
		    input->data(), output->data(), input->size());                     \
		break;

#define CALL_KERNEL_INPUT_TYPE_IMPLEMENTATION(INPUT_SHOGUN_TYPE)               \
	switch (output->get_type())                                                \
	{                                                                          \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::BOOLEAN)                          \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::INT8)                             \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::INT16)                            \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::INT32)                            \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::INT64)                            \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::UINT8)                            \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::UINT16)                           \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::UINT32)                           \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::UINT64)                           \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::FLOAT32)                          \
		CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION(                                \
		    INPUT_SHOGUN_TYPE, element_type::FLOAT64)                          \
	}

#define CALL_KERNEL(INPUT_SHOGUN_TYPE)                                         \
	case INPUT_SHOGUN_TYPE:                                                    \
	{                                                                          \
		CALL_KERNEL_INPUT_TYPE_IMPLEMENTATION(INPUT_SHOGUN_TYPE)               \
	}                                                                          \
	break;

					switch (input->get_type())
					{
						CALL_KERNEL(element_type::BOOLEAN)
						CALL_KERNEL(element_type::INT8)
						CALL_KERNEL(element_type::INT16)
						CALL_KERNEL(element_type::INT32)
						CALL_KERNEL(element_type::INT64)
						CALL_KERNEL(element_type::UINT8)
						CALL_KERNEL(element_type::UINT16)
						CALL_KERNEL(element_type::UINT32)
						CALL_KERNEL(element_type::UINT64)
						CALL_KERNEL(element_type::FLOAT32)
						CALL_KERNEL(element_type::FLOAT64)
					}
#undef CALL_KERNEL
#undef CALL_KERNEL_OUTPUT_TYPE_IMPLEMENTATION
#undef CALL_KERNEL_INPUT_TYPE_IMPLEMENTATION
				}

				template <typename InputType, typename OutputType>
				void
				kernel_implementation(void* input, void* output, size_t size)
				{
					std::transform(
					    static_cast<const InputType*>(input),
					    static_cast<const InputType*>(input) + size,
					    static_cast<OutputType*>(output),
					    [](const InputType& el) {
						    return static_cast<OutputType>(el);
					    });
				}
			}; // namespace op
		}      // namespace op
	}          // namespace graph
} // namespace shogun

#endif
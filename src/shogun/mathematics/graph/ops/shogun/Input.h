/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNINPUTSHOGUN_H_
#define SHOGUNINPUTSHOGUN_H_

#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/ops/abstract/Operator.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{

			IGNORE_IN_CLASSLIST class InputShogun
			    : public Operator
			{
			public:
				InputShogun(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				std::vector<std::shared_ptr<Tensor>>
				evaluate_input(const std::shared_ptr<Tensor>& tensor)
				{
					runtime_checks_and_allocation(std::vector{tensor});
					// to copy or not to copy?
					// m_output_tensor->allocate_tensor(tensor->get_shape());
					// m_output_tensor->data() = memcpy(...);
					m_output_tensors[0]->data() = tensor->data();
					return m_output_tensors;
				}

				std::string_view get_operator_name() const override
				{
					return "Input";
				}

				void call(const std::vector<
				          std::shared_ptr<detail::shogun::OutputNode>>&) final
				{
					error("Input nodes cannot be run with evaluate. Use "
					      "evaluate_input(Tensor) instead");
				}

			protected:
				void runtime_checks_and_allocation(
				    const std::vector<std::shared_ptr<Tensor>>& tensors) final
				{
					if (tensors.size() != 1)
						error("Input operation expected one input.");
					if (m_output_tensors.size() != 1)
						error("Input operation expected one output.");

					const auto& input_tensor = tensors[0];
					auto& output_tensor = m_output_tensors[0];

					runtime_type_check(input_tensor, output_tensor);
					runtime_shape_check(input_tensor, output_tensor);
				}

			private:
				void runtime_type_check(
				    const std::shared_ptr<Tensor>& input_tensor,
				    const std::shared_ptr<Tensor>& output_tensor)
				{
					if (input_tensor->get_type() != output_tensor->get_type())
						error("Input node got wrong input type!");
				}

				void runtime_shape_check(
				    const std::shared_ptr<Tensor>& input_tensor,
				    std::shared_ptr<Tensor>& output_tensor)
				{
					const auto& shape = input_tensor->get_shape();
					output_tensor->set_shape(shape);
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
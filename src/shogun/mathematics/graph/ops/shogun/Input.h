/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNINPUTSHOGUN_H_
#define SHOGUNINPUTSHOGUN_H_

#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/ops/abstract/Operator.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			/* The InputShogun operation transfers the input memory to the
			 * entry point of the DAG.
			 * Currently only CPU-CPU mapping is implemented
			 */
			IGNORE_IN_CLASSLIST class InputShogun : public Operator
			{
			public:
				InputShogun(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				std::vector<std::shared_ptr<ShogunStorage>>
				evaluate_input(const std::shared_ptr<Tensor>& tensor)
				{
					if (m_outputs.size() != 1)
						error("Input operation expected one output.");

					runtime_checks_and_allocation(tensor);
					return m_outputs;
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
				    const std::shared_ptr<Tensor>& input_tensor)
				{
					runtime_type_check(input_tensor, m_outputs[0]);
					runtime_shape_check(input_tensor, m_outputs[0]);
				}

			private:
				void runtime_type_check(
				    const std::shared_ptr<Tensor>& input_tensor,
				    const std::shared_ptr<ShogunStorage>& output)
				{
					if (input_tensor->get_type() != output->get_type())
						error("Input node got wrong input type!");
				}

				void runtime_shape_check(
				    const std::shared_ptr<Tensor>& input_tensor,
				    std::shared_ptr<ShogunStorage>& output)
				{
					// get copy of shared_ptr of Storage
					output = input_tensor->data();
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
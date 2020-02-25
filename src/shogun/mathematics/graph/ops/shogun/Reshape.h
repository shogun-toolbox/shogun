/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_RESHAPE_SHOGUN_H_
#define SHOGUN_RESHAPE_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Reshape.h>
#include <shogun/mathematics/graph/ops/abstract/Operator.h>


namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class ReshapeShogun : public Operator
			{
			public:
				ReshapeShogun(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Reshape";
				}

				void call(const std::vector<std::shared_ptr<
				              detail::shogun::OutputNode>>& input_nodes) final
				{
					const auto& input_tensor1 =
					    input_nodes[0]->get_output_tensors()[0];
					const auto& output_tensor = m_output_tensors[0];

					runtime_checks_and_allocation(
					    std::vector{input_tensor1});
				}

			private:
				void runtime_checks_and_allocation(
				    const std::vector<std::shared_ptr<Tensor>>& tensors) final
				{
					m_output_tensors[0]->data() = tensors[0]->data();
				}
			};
		}
	}
}

#endif
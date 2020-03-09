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
					const auto& input =
					    input_nodes[0]->get_outputs()[0];
					const auto final_shape = m_outputs[0]->get_shape();

					m_outputs[0] = std::move(input);
					m_outputs[0]->reshape(final_shape);
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
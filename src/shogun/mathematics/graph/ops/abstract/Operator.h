/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_GRAPH_OPERATOR_H_
#define SHOGUN_GRAPH_OPERATOR_H_

#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/nodes/Node.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				class OutputNode;
			}
		} // namespace detail

		namespace op
		{
			class Operator
			{
			public:
				Operator(const std::shared_ptr<node::Node>& node) : m_node(node)
				{
					const auto& shapes = m_node->get_shapes();
					const auto& types = m_node->get_types();

					for (const auto& [shape, type] :
					     zip_iterator(shapes, types))
					{
						m_output_tensors.push_back(
						    std::make_shared<Tensor>(shape, type));
					}
				}

				virtual ~Operator()
				{
				}

				const std::vector<std::shared_ptr<Tensor>>&
				operator()(const std::vector<
				           std::shared_ptr<detail::shogun::OutputNode>>& inputs)
				{
					call(inputs);
					return m_output_tensors;
				}

				virtual std::string_view get_operator_name() const = 0;

			protected:
				void virtual call(
				    const std::vector<
				        std::shared_ptr<detail::shogun::OutputNode>>&) = 0;

			protected:
				std::shared_ptr<node::Node> m_node;
				std::vector<std::shared_ptr<Tensor>> m_output_tensors;
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
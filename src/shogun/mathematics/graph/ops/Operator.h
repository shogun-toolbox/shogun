#include <shogun/mathematics/graph/Tensor.h>

#ifndef SHOGUN_GRAPH_OPERATOR_H_
#define SHOGUN_GRAPH_OPERATOR_H_

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
		}

		namespace op
		{
			class Operator
			{
			public:
				Operator(const std::shared_ptr<node::Node>& node): m_node(node)
				{
					const auto& shapes = m_node->get_shapes();
					const auto& types = m_node->get_types();

					for (const auto& [shape, type]: zip_iterator(shapes, types))
					{
						m_output_tensors.push_back(std::make_shared<Tensor>(shape, type));
					}
				}

				virtual ~Operator()
				{
				}

				const std::vector<std::shared_ptr<Tensor>>& operator()(const std::vector<std::shared_ptr<detail::shogun::OutputNode>>& inputs)
				{
					call(inputs);
					return m_output_tensors;
				}

				virtual std::string_view get_operator_name() const = 0;

			protected:
				void virtual call(const std::vector<std::shared_ptr<detail::shogun::OutputNode>>&) = 0;

				virtual void runtime_checks_and_allocation(const std::vector<std::shared_ptr<Tensor>>&) = 0;

			protected:
				std::shared_ptr<node::Node> m_node;
				std::vector<std::shared_ptr<Tensor>> m_output_tensors;

			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
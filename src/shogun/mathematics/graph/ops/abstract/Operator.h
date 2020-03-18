/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_GRAPH_OPERATOR_H_
#define SHOGUN_GRAPH_OPERATOR_H_

#include "shogun/mathematics/graph/Storage.h"
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
				Operator(const std::shared_ptr<node::Node>& node);

				virtual ~Operator() = default;

				const std::vector<std::shared_ptr<Storage>>&
				operator()(const std::vector<
				           std::shared_ptr<detail::shogun::OutputNode>>& inputs)
				{
					call(inputs);
					return m_outputs;
				}

				virtual std::string_view get_operator_name() const = 0;

			protected:
				virtual void
				call(const std::vector<
				     std::shared_ptr<detail::shogun::OutputNode>>&) = 0;

			protected:
				std::shared_ptr<node::Node> m_node;
				std::vector<std::shared_ptr<Storage>> m_outputs;
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif
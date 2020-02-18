/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DETAIL_DIVIDE_SHOGUN_H_
#define SHOGUN_DETAIL_DIVIDE_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Divide.h>
#include <shogun/mathematics/graph/ops/shogun/Divide.h>
#include <shogun/mathematics/graph/runtime/shogun/ShogunBinaryNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				IGNORE_IN_CLASSLIST class DivideShogun
				    : public ShogunBinaryRuntimeNode<
				          DivideShogun, node::Divide, OutputNode>
				{
				public:
					DivideShogun() : ShogunBinaryRuntimeNode()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Divide";
					}

					[[nodiscard]] std::shared_ptr<OutputNode>
					build_implementation_(
					    const std::shared_ptr<OutputNode>& node1,
					    const std::shared_ptr<OutputNode>& node2,
					    const std::shared_ptr<node::Node>& graph_node) const {
						return std::make_shared<OutputNode>(
						    std::make_shared<op::DivideShogun>(graph_node),
						    node1, node2);
					}
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

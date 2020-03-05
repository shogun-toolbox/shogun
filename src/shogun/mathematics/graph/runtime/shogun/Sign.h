/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DETAIL_SIGN_SHOGUN_H_
#define SHOGUN_DETAIL_SIGN_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Sign.h>
#include <shogun/mathematics/graph/ops/shogun/Sign.h>
#include <shogun/mathematics/graph/runtime/shogun/ShogunUnaryNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				IGNORE_IN_CLASSLIST class SignShogun
				    : public ShogunUnaryRuntimeNode<
				          SignShogun, node::Sign, OutputNode>
				{
				public:
					SignShogun() : ShogunUnaryRuntimeNode()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Sign";
					}

					[[nodiscard]] std::shared_ptr<OutputNode>
					build_implementation_(
					    const std::shared_ptr<OutputNode>& node,
					    const std::shared_ptr<node::Node>& graph_node) const {
						return std::make_shared<OutputNode>(
						    std::make_shared<op::SignShogun>(graph_node), node);
					}
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

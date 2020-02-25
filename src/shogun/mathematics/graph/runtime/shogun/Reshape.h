/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DETAIL_RESHAPE_SHOGUN_H_
#define SHOGUN_DETAIL_RESHAPE_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Reshape.h>
#include <shogun/mathematics/graph/ops/shogun/Reshape.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				IGNORE_IN_CLASSLIST class ReshapeShogun
				    : public RuntimeNodeTemplate<node::Reshape, OutputNode>
				{
				public:
					ReshapeShogun() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Reshape";
					}

					[[nodiscard]] std::shared_ptr<OutputNode>
					build_implementation(const std::shared_ptr<node::Node>&
					                         graph_node) const final {
						if (this->m_input_nodes.size() != 1)
							error(
							    "Expected one input node in a dot operation.");

						const auto& input_node1 = this->m_input_nodes[0];

						return std::make_shared<OutputNode>(
						    std::make_shared<op::ReshapeShogun>(graph_node),
						    input_node1);
					}
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

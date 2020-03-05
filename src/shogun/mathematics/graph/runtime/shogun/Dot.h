/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DETAIL_DOT_SHOGUN_H_
#define SHOGUN_DETAIL_DOT_SHOGUN_H_

#include "OutputNode.h"
#include <shogun/mathematics/graph/nodes/Dot.h>
#include <shogun/mathematics/graph/ops/shogun/Dot.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				IGNORE_IN_CLASSLIST class DotShogun
				    : public RuntimeNodeTemplate<node::Dot, OutputNode>
				{
				public:
					DotShogun() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Dot";
					}

					[[nodiscard]] std::shared_ptr<OutputNode>
					build_implementation(const std::shared_ptr<node::Node>&
					                         graph_node) const final {
						if (this->m_input_nodes.size() != 2)
							error(
							    "Expected two input nodes in a dot operation.");

						const auto& input_node1 = this->m_input_nodes[0];
						const auto& input_node2 = this->m_input_nodes[1];

						return std::make_shared<OutputNode>(
						    std::make_shared<op::DotShogun>(graph_node),
						    input_node1, input_node2);
					}
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

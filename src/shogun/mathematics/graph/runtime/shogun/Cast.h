/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DETAIL_CAST_SHOGUN_H_
#define SHOGUN_DETAIL_CAST_SHOGUN_H_

#include "OutputNode.h"
#include <shogun/mathematics/graph/ops/shogun/Cast.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				IGNORE_IN_CLASSLIST class CastShogun
				    : public RuntimeNodeTemplate<node::Cast, OutputNode>
				{
				public:
					CastShogun() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Cast";
					}

					[[nodiscard]] std::shared_ptr<OutputNode>
					build_implementation(const std::shared_ptr<node::Node>&
					                         graph_node) const final {
						if (this->m_input_nodes.size() != 1)
							error("Expected one input node in cast operation.");

						const auto& input_node1 = this->m_input_nodes[0];

						return std::make_shared<OutputNode>(
						    std::make_shared<op::CastShogun>(graph_node),
						    input_node1);
					}
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

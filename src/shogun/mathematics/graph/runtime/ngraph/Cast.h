/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_CAST_NGRAPH_H_
#define SHOGUN_CAST_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Cast.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>
#include <shogun/mathematics/graph/runtime/ngraph/Input.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class CastNGraph
				    : public RuntimeNodeTemplate<node::Cast, ::ngraph::Node>
				{
				public:
					CastNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Cast";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						if (m_input_nodes.size() != 1)
							error("Expected once node in "
							      "CastNGraph.");
						const auto& type = node->get_types()[0];

						return std::make_shared<::ngraph::op::Convert>(
						    m_input_nodes[0], get_ngraph_type_from_enum(type));
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

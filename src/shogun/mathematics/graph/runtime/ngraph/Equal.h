/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_EQUAL_NGRAPH_H_
#define SHOGUN_EQUAL_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Equal.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

#include <ngraph/op/equal.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class EqualNGraph

				    : public RuntimeNodeTemplate<node::Equal, ::ngraph::Node>
				{
				public:
					EqualNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Equal";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						if (m_input_nodes.size() != 2)
							error("Expected two input nodes in "
							      "EqualNGraph.");
						return std::make_shared<::ngraph::op::Equal>(
						    m_input_nodes[0], m_input_nodes[1]);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_LOGICAL_AND_NGRAPH_H_
#define SHOGUN_LOGICAL_AND_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/LogicalAnd.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

#include <ngraph/op/and.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class LogicalAndNGraph

				    : public RuntimeNodeTemplate<node::LogicalAnd, ::ngraph::Node>
				{
				public:
					LogicalAndNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "LogicalAnd";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						if (m_input_nodes.size() != 2)
							error("Expected two input nodes in "
							      "LogicalAndNGraph.");
						return std::make_shared<::ngraph::op::And>(
						    m_input_nodes[0], m_input_nodes[1]);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

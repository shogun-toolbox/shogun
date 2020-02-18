/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_LOGICAL_XOR_NGRAPH_H_
#define SHOGUN_LOGICAL_XOR_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/LogicalXor.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

#include <ngraph/op/or.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class LogicalXorNGraph

				    : public RuntimeNodeTemplate<
				          node::LogicalXor, ::ngraph::Node>
				{
				public:
					LogicalXorNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "LogicalXor";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						if (m_input_nodes.size() != 2)
							error("Expected two input nodes in "
							      "LogicalXor.");
						return std::make_shared<::ngraph::op::Xor>(
						    m_input_nodes[0], m_input_nodes[1]);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

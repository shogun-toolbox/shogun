/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_EXP_NGRAPH_H_
#define SHOGUN_EXP_NGRAPH_H_

#include "Input.h"
#include <shogun/mathematics/graph/nodes/Exp.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

#include <ngraph/op/exp.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class ExpNGraph

				    : public RuntimeNodeTemplate<node::Exp, ::ngraph::Node>
				{
				public:
					ExpNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Exp";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						if (m_input_nodes.size() != 1)
							error("Expected one input node in ExpNGraph.");
						return std::make_shared<::ngraph::op::Exp>(
						    m_input_nodes[0]);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

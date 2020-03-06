/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_SIGN_NGRAPH_H_
#define SHOGUN_SIGN_NGRAPH_H_

#include "Input.h"
#include <ngraph/op/sign.hpp>
#include <shogun/mathematics/graph/nodes/Sign.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class SignNGraph

				    : public RuntimeNodeTemplate<node::Sign, ::ngraph::Node>
				{
				public:
					SignNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Sign";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						if (m_input_nodes.size() != 1)
							error("Expected one input node in SignNGraph.");
						return std::make_shared<::ngraph::op::Sign>(
						    m_input_nodes[0]);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

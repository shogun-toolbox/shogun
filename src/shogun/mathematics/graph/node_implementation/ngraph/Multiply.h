/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_MULTIPLY_NGRAPH_H_
#define SHOGUN_MULTIPLY_NGRAPH_H_

#include <shogun/mathematics/graph/node_implementation/NodeImplementation.h>
#include <shogun/mathematics/graph/nodes/Multiply.h>

#include <ngraph/op/multiply.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class MultiplyNGraph

				    : public RuntimeNodeTemplate<node::Multiply, ::ngraph::Node>
				{
				public:
					MultiplyNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Multiply";
					}

					std::shared_ptr<::ngraph::Node> build_implementation(
					    const std::shared_ptr<node::Node>& node) const final
					{
						if (m_input_nodes.size() != 2)
							error(
							    "Expected two input nodes in MultiplyNGraph.");
						return std::make_shared<::ngraph::op::Multiply>(
						    m_input_nodes[0], m_input_nodes[1]);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

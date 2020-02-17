/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ADD_NGRAPH_H_
#define SHOGUN_ADD_NGRAPH_H_

#include <shogun/mathematics/graph/node_implementation/NodeImplementation.h>
#include <shogun/mathematics/graph/nodes/Add.h>

#include <ngraph/op/add.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class AddNGraph

			: public RuntimeNodeTemplate<node::Add, ::ngraph::Node>
				{
				public:
					AddNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Add";
					}

					std::shared_ptr<::ngraph::Node>
					build_implementation(const std::shared_ptr<node::Node>& node) const final
					{
						if (m_input_nodes.size() != 2)
							error("Expected two input nodes in "
							      "AddNGraph.");
						return std::make_shared<::ngraph::op::Add>(
							    m_input_nodes[0], m_input_nodes[1]);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

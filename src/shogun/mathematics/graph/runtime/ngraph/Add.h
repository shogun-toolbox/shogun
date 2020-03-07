/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ADD_NGRAPH_H_
#define SHOGUN_ADD_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/runtime/ngraph/BinaryRuntimeNode.h>

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
				    : public BinaryRuntimeNodeNGraph<
				          node::Add, ::ngraph::op::Add>
				{
				public:
					AddNGraph() : BinaryRuntimeNodeNGraph()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Add";
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

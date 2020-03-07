/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_LOGICAL_OR_NGRAPH_H_
#define SHOGUN_LOGICAL_OR_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/LogicalOr.h>
#include <shogun/mathematics/graph/runtime/ngraph/BinaryRuntimeNode.h>

#include <ngraph/op/or.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class LogicalOrNGraph
				    : public BinaryRuntimeNodeNGraph<
				          node::LogicalOr, ::ngraph::op::Or>
				{
				public:
					LogicalOrNGraph() : BinaryRuntimeNodeNGraph()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "LogicalOr";
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

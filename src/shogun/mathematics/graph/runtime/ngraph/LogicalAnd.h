/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_LOGICAL_AND_NGRAPH_H_
#define SHOGUN_LOGICAL_AND_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/LogicalAnd.h>
#include <shogun/mathematics/graph/runtime/ngraph/BinaryRuntimeNode.h>

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
				    : public BinaryRuntimeNodeNGraph<
				          node::LogicalAnd, ::ngraph::op::And>
				{
				public:
					LogicalAndNGraph() : BinaryRuntimeNodeNGraph()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "LogicalAnd";
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

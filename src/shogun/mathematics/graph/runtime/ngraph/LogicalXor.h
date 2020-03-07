/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_LOGICAL_XOR_NGRAPH_H_
#define SHOGUN_LOGICAL_XOR_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/LogicalXor.h>
#include <shogun/mathematics/graph/runtime/ngraph/BinaryRuntimeNode.h>

#include <ngraph/op/xor.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class LogicalXorNGraph
				    : public BinaryRuntimeNodeNGraph<
				          node::LogicalXor, ::ngraph::op::Xor>
				{
				public:
					LogicalXorNGraph() : BinaryRuntimeNodeNGraph()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "LogicalXor";
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_SUBTRACT_NGRAPH_H_
#define SHOGUN_SUBTRACT_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Subtract.h>
#include <shogun/mathematics/graph/runtime/ngraph/BinaryRuntimeNode.h>

#include <ngraph/op/subtract.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class SubtractNGraph
				    : public BinaryRuntimeNodeNGraph<
				          node::Subtract, ::ngraph::op::Subtract>
				{
				public:
					SubtractNGraph() : BinaryRuntimeNodeNGraph()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Subtract";
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

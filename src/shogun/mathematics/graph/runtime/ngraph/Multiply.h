/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_MULTIPLY_NGRAPH_H_
#define SHOGUN_MULTIPLY_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Multiply.h>
#include <shogun/mathematics/graph/runtime/ngraph/BinaryRuntimeNode.h>

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
				    : public BinaryRuntimeNodeNGraph<
				          node::Multiply, ::ngraph::op::Multiply>
				{
				public:
					MultiplyNGraph() : BinaryRuntimeNodeNGraph()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Multiply";
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

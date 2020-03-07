/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DIVIDE_NGRAPH_H_
#define SHOGUN_DIVIDE_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Divide.h>
#include <shogun/mathematics/graph/runtime/ngraph/BinaryRuntimeNode.h>

#include <ngraph/op/divide.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class DivideNGraph
				    : public BinaryRuntimeNodeNGraph<
				          node::Divide, ::ngraph::op::Divide>
				{
				public:
					DivideNGraph() : BinaryRuntimeNodeNGraph()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Divide";
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

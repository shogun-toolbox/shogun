/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_EQUAL_NGRAPH_H_
#define SHOGUN_EQUAL_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Equal.h>
#include <shogun/mathematics/graph/runtime/ngraph/BinaryRuntimeNode.h>

#include <ngraph/op/equal.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class EqualNGraph
				    : public BinaryRuntimeNodeNGraph<node::Equal, ::ngraph::op::Equal>
				{
				public:
					EqualNGraph() : BinaryRuntimeNodeNGraph()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Equal";
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

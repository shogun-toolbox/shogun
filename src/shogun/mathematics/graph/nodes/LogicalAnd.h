/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODE_LOGICAL_AND_H_
#define SHOGUN_NODE_LOGICAL_AND_H_

#include <shogun/mathematics/graph/nodes/LogicalBinaryNode.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class LogicalAnd : public LogicalBinaryNode
			{
			public:
				LogicalAnd(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				    : LogicalBinaryNode(node1, node2)
				{
				}

				std::string to_string() const final
				{
					return fmt::format(
					    "LogicalAnd(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const final
				{
					return "LogicalAnd";
				}
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
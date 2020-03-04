/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODE_EQUAL_H_
#define SHOGUN_NODE_EQUAL_H_

#include <shogun/mathematics/graph/nodes/BinaryNode.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class Equal : public BinaryNode
			{
			public:
				Equal(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				    : BinaryNode(node1, node2)
				{
					// output of equal is boolean
					m_types[0] = element_type::BOOLEAN;
				}

				std::string to_string() const override
				{
					return fmt::format(
					    "Equal(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const override
				{
					return "Equal";
				}
			};

		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
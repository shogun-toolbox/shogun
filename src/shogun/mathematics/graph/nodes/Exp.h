/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODE_EXP_H_
#define SHOGUN_NODE_EXP_H_

#include <shogun/mathematics/graph/nodes/UnaryNode.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class Exp : public UnaryNode
			{
			public:
				Exp(const std::shared_ptr<Node>& node) : UnaryNode(node)
				{
				}

				std::string to_string() const override
				{
					return fmt::format(
					    "Exp(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const override
				{
					return "Exp";
				}
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
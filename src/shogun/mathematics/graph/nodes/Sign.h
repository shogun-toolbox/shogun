/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_NODE_SIGN_H_
#define SHOGUN_NODE_SIGN_H_

#include <shogun/mathematics/graph/nodes/UnaryNode.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class Sign : public UnaryNode
			{
			public:
				Sign(const std::shared_ptr<Node>& node) : UnaryNode(node)
				{
					const auto& type = m_types[0];
					if (is_unsigned(type))
						error(
						    "Sign does not work with unsigned operators. "
						    "Sign got input node: {}",
						    node->to_string());
				}

				std::string to_string() const override
				{
					return fmt::format(
					    "Sign(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const override
				{
					return "Sign";
				}
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
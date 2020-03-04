/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSORSUBTRACT_H_
#define SHOGUNTENSORSUBTRACT_H_

#include <shogun/mathematics/graph/nodes/BinaryNode.h>

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class Subtract : public BinaryNode
			{
			public:
				Subtract(
				    const std::shared_ptr<Node>& node1,
				    const std::shared_ptr<Node>& node2)
				    : BinaryNode(node1, node2)
				{
				}

				std::string to_string() const override
				{
					return fmt::format(
					    "Subtract(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}

				std::string_view get_node_name() const override
				{
					return "Subtract";
				}
			};

			std::shared_ptr<Node> operator-(
			    const std::shared_ptr<Node>& A, const std::shared_ptr<Node>& B)
			{
				return std::make_shared<Subtract>(A, B);
			}
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
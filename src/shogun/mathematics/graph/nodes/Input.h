/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNINPUT_H_
#define SHOGUNINPUT_H_

#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/nodes/Node.h>

#include <shogun/util/enumerate.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			IGNORE_IN_CLASSLIST class Input : public Node
			{
			public:
				Input(const Shape& shape, element_type type) : Node(shape, type)
				{
				}

				std::string_view get_node_name() const final
				{
					return "Input";
				}

				std::string to_string() const final
				{
					return fmt::format(
					    "Input(shape={}, type={})", get_shapes()[0],
					    get_types()[0]);
				}
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun

#endif
/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_RESHAPE_NGRAPH_H_
#define SHOGUN_RESHAPE_NGRAPH_H_

#include "Input.h"
#include <ngraph/op/reshape.hpp>
#include <shogun/mathematics/graph/nodes/Reshape.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				IGNORE_IN_CLASSLIST class ReshapeNGraph

				    : public RuntimeNodeTemplate<node::Reshape, ::ngraph::Node>
				{
				public:
					ReshapeNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Reshape";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						if (m_input_nodes.size() != 1)
							error("Expected two input nodes in "
							      "AddNGraph.");
						const auto& input_shape =
						    node->get_input_nodes()[0]->get_shapes()[0];
						const auto& output_shape = node->get_shapes()[0];
						std::vector<size_t> axis_vector(input_shape.size());
						std::iota(axis_vector.begin(), axis_vector.end(), 0);
						return std::make_shared<::ngraph::op::Reshape>(
						    m_input_nodes[0], ::ngraph::AxisVector{axis_vector},
						    to_ngraph_shape(output_shape));
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

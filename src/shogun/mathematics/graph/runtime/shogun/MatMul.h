/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DETAIL_MATMUL_SHOGUN_H_
#define SHOGUN_DETAIL_MATMUL_SHOGUN_H_

#include "OutputNode.h"
#include <shogun/mathematics/graph/nodes/MatMul.h>
#include <shogun/mathematics/graph/ops/shogun/MatMul.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				IGNORE_IN_CLASSLIST class MatMulShogun
				    : public RuntimeNodeTemplate<node::MatMul, OutputNode>
				{
				public:
					MatMulShogun() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "MatMul";
					}

					[[nodiscard]] std::shared_ptr<OutputNode>
					build_implementation(const std::shared_ptr<node::Node>&
					                         graph_node) const final {
						const auto input_node =
						    std::static_pointer_cast<node::MatMul>(graph_node);
						if (this->m_input_nodes.size() != 2)
							error("Expected two input nodes in "
							      "MatMulShogun.");

						return std::make_shared<OutputNode>(
						    std::make_shared<op::MatMulShogun>(input_node),
						    this->m_input_nodes[0], this->m_input_nodes[1]);
					}
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

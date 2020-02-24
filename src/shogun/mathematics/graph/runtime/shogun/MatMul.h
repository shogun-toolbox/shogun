/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_MATMUL_SHOGUN_H_
#define SHOGUN_MATMUL_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/MatMul.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>
#include <shogun/mathematics/graph/ops/shogun/MatMul.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
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

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_implementation(
					    const std::shared_ptr<node::Node>& node) const final {
						const auto input_node =
						    std::static_pointer_cast<node::MatMul>(node);
						if (input_node.size() != 2)
							error("Expected two input nodes in "
							      "MatMulShogun.");
						const bool transpose_a = input_node->get_transpose_a();
						const bool transpose_b = input_node->get_transpose_b();

						return std::make_shared<op::MatMul>(
						    input_node[0], input_node[1], transpose_a,
						    transpose_b);
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif

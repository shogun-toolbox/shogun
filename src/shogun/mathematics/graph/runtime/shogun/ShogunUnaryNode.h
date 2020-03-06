/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_UNARY_NODE_IMPL_H_
#define SHOGUN_UNARY_NODE_IMPL_H_

#include <shogun/mathematics/graph/runtime/RuntimeNode.h>
#include <shogun/mathematics/graph/runtime/shogun/OutputNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				template <
				    typename RuntimeNodeType, typename AbstractNodeType,
				    typename InterfaceOperator>
				class ShogunUnaryRuntimeNode
				    : public RuntimeNodeTemplate<
				          AbstractNodeType, InterfaceOperator>
				{
				public:
					ShogunUnaryRuntimeNode()
					    : RuntimeNodeTemplate<
					          AbstractNodeType, InterfaceOperator>()
					{
					}

					virtual ~ShogunUnaryRuntimeNode(){}

					    [[nodiscard]] std::
					        shared_ptr<InterfaceOperator> build_implementation(
					            const std::shared_ptr<node::Node>& node)
					            const final
					{
						if (this->m_input_nodes.size() != 1)
							error("Expected one input node in a binary "
							      "operation.");

						const auto& input_node = this->m_input_nodes[0];

						return static_cast<const RuntimeNodeType*>(this)
						    ->build_implementation_(input_node, node);
					}
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif
#ifndef SHOGUN_BINARY_NODE_IMPL_H_ 
#define SHOGUN_BINARY_NODE_IMPL_H_

#include <shogun/mathematics/graph/node_implementation/shogun/OutputNode.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun {
				template <
				    typename RuntimeNodeType, typename AbstractNodeType,
				    typename InterfaceOperator>
				class ShogunBinaryRuntimeNode
				    : public RuntimeNodeTemplate<
				          AbstractNodeType, InterfaceOperator>
				{
				public:
					ShogunBinaryRuntimeNode()
					    : RuntimeNodeTemplate<AbstractNodeType, InterfaceOperator>()
					{
					}

					virtual ~ShogunBinaryRuntimeNode()
					{
					}

					[[nodiscard]] std::shared_ptr<InterfaceOperator> build_implementation(const std::shared_ptr<node::Node>& node) const final
					{
						if (this->m_input_nodes.size() != 2)
							error("Expected two input nodes in a binary operation.");

						const auto& input_node1 = this->m_input_nodes[0];
						const auto& input_node2 = this->m_input_nodes[1];

						return static_cast<const RuntimeNodeType*>(this)->build_implementation_(input_node1, input_node2, node);
					}
				};
			}
		} // namespace detail
	}     // namespace graph
} // namespace shogun

#endif
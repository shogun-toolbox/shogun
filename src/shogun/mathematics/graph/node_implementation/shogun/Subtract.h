// /*
//  * This software is distributed under BSD 3-clause license (see LICENSE file).
//  *
//  * Authors: Gil Hoben
//  */

// #ifndef SHOGUN_DETAIL_SUBTRACTSHOGUN_H_
// #define SHOGUN_DETAIL_SUBTRACTSHOGUN_H_

// #include <shogun/mathematics/graph/node_implementation/shogun/ShogunBinaryNode.h>
// #include <shogun/mathematics/graph/nodes/Subtract.h>
// #include <shogun/mathematics/graph/ops/shogun/Subtract.h>

// namespace shogun
// {
// 	namespace graph
// 	{
// 		namespace detail
// 		{
// 			namespace shogun {
// 				IGNORE_IN_CLASSLIST class SubtractShogun
// 				    : public ShogunBinaryRuntimeNode<SubtractShogun, node::Subtract, OutputNode>
// 				{
// 				public:
// 					SubtractShogun() : ShogunBinaryRuntimeNode()
// 					{
// 					}

// 					std::string_view get_runtime_node_name() const final
// 					{
// 						return "Subtract";
// 					}

// 					[[nodiscard]] std::shared_ptr<OutputNode> build_implementation_(
// 						const std::shared_ptr<OutputNode>& node1,
// 						const std::shared_ptr<OutputNode>& node2,
// 						const std::shared_ptr<node::Node>& graph_node) const
// 					{
// 					    return std::make_shared<OutputNode>(
// 					    	std::make_shared<op::SubtractShogun>(node1, node2, 
// 					    		graph_node->get_shapes(), graph_node->get_types()));
// 					}
// 				};
// 			}
// 		} // namespace detail
// 	}     // namespace graph
// } // namespace shogun

// #endif

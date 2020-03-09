/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OUTPUT_NODE_H_
#define SHOGUN_OUTPUT_NODE_H_

#include <shogun/mathematics/graph/ops/abstract/Operator.h>
#include <shogun/mathematics/graph/ops/abstract/ShogunStorage.h>
#include <shogun/mathematics/graph/ops/shogun/Input.h>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				/* A OutputNode stores and passes the input nodes required
				 * by an operation and holds a pointer to the resulting
				 * allocated results. These can then passed on to the
				 * next OutputNode in the graph.
				 */
				class OutputNode
				{
				public:
					OutputNode(const std::shared_ptr<op::InputShogun>& op)
					    : m_op(op)
					{
					}

					template <typename... Args>
					OutputNode(
					    const std::shared_ptr<op::Operator>& op,
					    Args&&... nodes)
					    : m_op(op), m_input_nodes{nodes...}
					{
					}

					void operator()()
					{
						if (m_input_nodes.empty())
							error(
							    "Input nodes need to receive an input tensor. "
							    "Use OutputNode::evaluate_tensor(Tensor) "
							    "instead");
						else
							m_outputs = m_op->operator()(m_input_nodes);
					}

					/* This member functions is only called by input nodes,
					 * and is the entry point of the execution graph.
					 */
					void evaluate_tensor(const std::shared_ptr<Tensor>& tensor)
					{
						m_outputs =
						    std::static_pointer_cast<op::InputShogun>(m_op)
						        ->evaluate_input(tensor);
					}

					const std::vector<std::shared_ptr<ShogunStorage>>&
					get_outputs() const
					{
						return m_outputs;
					}

					const std::vector<std::shared_ptr<OutputNode>>&
					get_input_nodes() const
					{
						return m_input_nodes;
					}

				private:
					/* The actual implementation of the operation */
					std::shared_ptr<op::Operator> m_op;
					/* Reference to inputs */
					std::vector<std::shared_ptr<OutputNode>> m_input_nodes;
					/* Reference to Operator allocated output storage */
					std::vector<std::shared_ptr<ShogunStorage>> m_outputs;
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif
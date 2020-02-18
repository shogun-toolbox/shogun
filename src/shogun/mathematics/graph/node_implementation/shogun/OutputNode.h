#include <shogun/mathematics/graph/ops/Operator.h>
#include <shogun/mathematics/graph/ops/shogun/Input.h>

#ifndef SHOGUN_OUTPUT_NODE_H_
#define SHOGUN_OUTPUT_NODE_H_

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
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
							m_output_tensors = m_op->operator()(m_input_nodes);
					}

					void evaluate_tensor(const std::shared_ptr<Tensor>& tensor)
					{
						m_output_tensors =
						    std::static_pointer_cast<op::InputShogun>(m_op)
						        ->evaluate_input(tensor);
					}

					const std::vector<std::shared_ptr<Tensor>>&
					get_output_tensors() const
					{
						return m_output_tensors;
					}

					const std::vector<std::shared_ptr<OutputNode>>&
					get_input_nodes() const
					{
						return m_input_nodes;
					}

				private:
					std::shared_ptr<op::Operator> m_op;
					std::vector<std::shared_ptr<OutputNode>> m_input_nodes;
					std::vector<std::shared_ptr<Tensor>> m_output_tensors;
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif
#ifndef SHOGUN_GRAPH_
#define SHOGUN_GRAPH_

#include <shogun/mathematics/graph/GraphExecutor.h>
#include <shogun/mathematics/graph/node_implementation/shogun/OutputNode.h>
#include <shogun/mathematics/graph/ops/Operator.h>

#include <memory>
#include <vector>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		IGNORE_IN_CLASSLIST class ShogunGraph : public GraphExecutor
		{
		public:
			static constexpr GRAPH_BACKEND kBackendType = GRAPH_BACKEND::SHOGUN;

			~ShogunGraph() override = default;
			std::vector<std::shared_ptr<Tensor>> execute(
			    const std::vector<std::shared_ptr<Tensor>>& tensors,
			    const std::vector<std::shared_ptr<node::Node>>& output_nodes)
			    const final;

			void
			add_input_operator(const std::shared_ptr<node::Node>& node) final;
			void
			add_operator_node(const std::shared_ptr<node::Node>& node) final;

		private:
			std::shared_ptr<Tensor>
			extract_result(const std::shared_ptr<node::Node>& node) const;
			std::shared_ptr<detail::RuntimeNode>
			get_operator(const std::shared_ptr<node::Node>& node) const;
			std::unordered_map<
			    std::shared_ptr<node::Node>,
			    std::shared_ptr<detail::shogun::OutputNode>>
			    m_lookup;
			std::vector<std::shared_ptr<detail::shogun::OutputNode>>
			    m_input_output_nodes;
			std::vector<std::shared_ptr<detail::shogun::OutputNode>>
			    m_operator_output_nodes;
		};
	} // namespace graph
} // namespace shogun

#endif /* SHOGUN_GRAPH_ */
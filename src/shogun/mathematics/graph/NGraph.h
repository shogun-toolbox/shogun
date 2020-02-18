#ifndef SHOGUN_GRAPH_
#define SHOGUN_GRAPH_

#include <shogun/mathematics/graph/GraphExecutor.h>

#include <ngraph/ngraph.hpp>

#include <memory>
#include <vector>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		IGNORE_IN_CLASSLIST class NGraph : public GraphExecutor
		{
		public:
			static constexpr GRAPH_BACKEND kBackendType = GRAPH_BACKEND::NGRAPH;

			~NGraph() override = default;
			std::vector<std::shared_ptr<Tensor>> execute(
			    const std::vector<std::shared_ptr<Tensor>>& tensors,
			    const std::vector<std::shared_ptr<node::Node>>&) const override;
			void add_input_operator(
			    const std::shared_ptr<node::Node>& node) override;
			void
			add_operator_node(const std::shared_ptr<node::Node>& node) override;

			std::shared_ptr<detail::RuntimeNode>
			get_operator(const std::shared_ptr<node::Node>& node) const;

		private:
			std::unordered_map<
			    std::shared_ptr<node::Node>, std::shared_ptr<ngraph::Node>>
			    m_lookup;

			std::vector<std::shared_ptr<ngraph::Node>> m_input_output_nodes;
			std::vector<std::shared_ptr<ngraph::Node>> m_operator_output_nodes;
		};
	} // namespace graph
} // namespace shogun

#endif /* SHOGUN_GRAPH_ */

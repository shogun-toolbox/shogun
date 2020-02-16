#ifndef GRAPH_
#define GRAPH_

#include <deque>
#include <memory>
#include <shogun/mathematics/graph/GraphExecutor.h>
#include <shogun/mathematics/graph/node_implementation/NodeImplementation.h>
#include <shogun/mathematics/graph/nodes/Input.h>

#include <unordered_set>
#include <vector>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{

		IGNORE_IN_CLASSLIST class Graph
		{
			enum class STATUS
			{
				UNMARKED = 0,
				TEMPORARY = 1,
				MARKED = 2
			};

		public:
			Graph(
			    const std::vector<std::shared_ptr<node::Input>>& inputs,
			    const std::vector<std::shared_ptr<node::Node>>& outputs);

			std::vector<std::shared_ptr<Tensor>>
			evaluate(const std::vector<std::shared_ptr<Tensor>>& tensors);

			void build();
			void build(GRAPH_BACKEND backend);

		private:
			std::unordered_map<std::shared_ptr<node::Node>, STATUS>
			check_fully_connected(
			    const std::vector<std::shared_ptr<node::Input>>& inputs,
			    const std::vector<std::shared_ptr<node::Node>>& outputs);
			void build_backend_graph(
			    std::unordered_map<std::shared_ptr<node::Node>, STATUS>&
			        unordered_nodes);
			void order_graph_visit_(
			    const std::shared_ptr<node::Node>& node,
			    std::unordered_map<std::shared_ptr<node::Node>, Graph::STATUS>&
			        all_nodes,
			    std::deque<std::shared_ptr<node::Node>>& result);

			std::vector<std::shared_ptr<node::Input>> m_inputs;
			std::vector<std::shared_ptr<node::Node>> m_outputs;

			std::deque<std::shared_ptr<node::Node>> m_cached_nodes;

			std::shared_ptr<GraphExecutor> m_executor;
		};
	} // namespace graph
} // namespace shogun

#endif /* GRAPH_ */

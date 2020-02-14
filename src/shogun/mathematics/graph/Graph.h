#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/mathematics/graph/nodes/Input.h>
#include <vector>
#include <deque>

#define IGNORE_IN_CLASSLIST

namespace shogun
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
		    const std::vector<std::shared_ptr<Input>>& inputs,
		    const std::vector<std::shared_ptr<Node>>& outputs);


		std::vector<std::shared_ptr<Tensor>> evaluate(const std::vector<std::shared_ptr<Tensor>>& tensors);
		
		void build();

#ifdef USE_NGRAPH
		std::shared_ptr<ngraph::Function> get_ngraph_function()
		{
		}
#endif
	private:
		std::unordered_map<std::shared_ptr<Node>, STATUS> check_fully_connected(
		    const std::vector<std::shared_ptr<Input>>& inputs,
		    const std::vector<std::shared_ptr<Node>>& outputs);
		void build_backend_graph(std::unordered_map<std::shared_ptr<Node>, STATUS>& unordered_nodes);
		void order_graph_visit_(const std::shared_ptr<Node>& node, 
			std::unordered_map<std::shared_ptr<Node>, Graph::STATUS>& all_nodes,
			std::deque<std::shared_ptr<Node>>& result);

		std::shared_ptr<Operator> add_operator_node(const std::shared_ptr<Node>&);

		void execute_shogun(const std::vector<std::shared_ptr<Tensor>>& tensors);
		void execute_ngraph(const std::vector<std::shared_ptr<Tensor>>& tensors);

		void build_shogun_graph();
		void build_ngraph_graph();


		std::vector<std::shared_ptr<Input>> m_inputs;
		std::vector<std::shared_ptr<Node>> m_outputs;

		std::deque<std::shared_ptr<Node>> m_cached_nodes;
		std::vector<std::shared_ptr<Input>> m_cached_input_nodes;
		std::vector<std::shared_ptr<Node>> m_cached_output_nodes;

		std::vector<std::shared_ptr<Operator>> m_cached_input_operators;
		std::vector<std::shared_ptr<Operator>> m_cached_operators;
    };

} // namespace shogun
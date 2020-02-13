#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/mathematics/graph/ops/Input.h>
#include <vector>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	IGNORE_IN_CLASSLIST class Graph
	{
	public:
		Graph(
		    const std::vector<std::shared_ptr<Input>>& inputs,
		    const std::vector<std::shared_ptr<Node>>& outputs);


		void evaluate(const std::vector<std::shared_ptr<Tensor>>& tensors);
		
		void build();

#ifdef USE_NGRAPH
		std::shared_ptr<ngraph::Function> get_ngraph_function()
		{
		}
#endif
	private:
		void check_fully_connected(
		    const std::vector<std::shared_ptr<Input>>& inputs,
		    const std::vector<std::shared_ptr<Node>>& outputs);
		void build_backend_graph(
		    const std::vector<std::shared_ptr<Input>>& inputs,
		    const std::vector<std::shared_ptr<Node>>& outputs);
		void get_backend_operator(const std::shared_ptr<Node>&);

		void execute_shogun();
		void execute_ngraph();

		void build_shogun_graph();
		void build_ngraph_graph();


		std::vector<std::shared_ptr<Input>> m_inputs;
		std::vector<std::shared_ptr<Node>> m_outputs;
	};

} // namespace shogun
#ifndef SHOGUN_GRAPH_
#define SHOGUN_GRAPH_

#include <shogun/mathematics/graph/GraphExecutor.h>

#include <memory>
#include <vector>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
    IGNORE_IN_CLASSLIST class NGraph: public GraphExecutor
    {
        public:
            static constexpr GRAPH_BACKEND kBackendType = GRAPH_BACKEND::NGRAPH;

            ~NGraph() override = default;
            void execute(const std::vector<std::shared_ptr<Tensor>>& tensors) const override;
            void add_input_operator(const std::shared_ptr<Node>& node) override;
            void add_operator_node(const std::shared_ptr<Node>& node) override;
    };
}

#endif /* SHOGUN_GRAPH_ */

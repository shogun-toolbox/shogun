#include <gtest/gtest.h>

#include "Graph.h"
#include "nodes/Add.h"
#include "nodes/Dot.h"
#include "nodes/Input.h"
#include "nodes/Multiply.h"
#include "nodes/Sign.h"
#include "nodes/Cast.h"
#include "nodes/Equal.h"

#include "test/GraphTest.h"

#include <shogun/features/DataGenerator.h>

#include <random>

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, perceptron_stochastic_gradient_descent)
{
#if 0
	using NumericType = typename TypeParam::c_type;

	if constexpr (!std::is_same_v<NumericType, float64_t>)
		return;
	else
	{
		size_t max_iters = 100; 
		int64_t num_feat = 5;
		int64_t num_examples = 1000;
		float64_t learning_rate = 0.001;


		auto X = make_shared<node::Input>(Shape{num_feat}, element_type::FLOAT64);
		auto y = make_shared<node::Input>(Shape{}, element_type::FLOAT64);
		auto w_input = make_shared<node::Input>(Shape{Shape::Dynamic}, element_type::FLOAT64);
		auto b_input = make_shared<node::Input>(Shape{}, element_type::FLOAT64);
		auto lr = make_shared<node::Input>(Shape{}, element_type::FLOAT64);

	    
	    auto predicted = make_shared<node::Dot>(X, w_input) + b_input;
	    auto correct = make_shared<node::Equal>(make_shared<node::Sign>(predicted), y);
	    auto gradient = make_shared<node::Cast>(correct, element_type::FLOAT64) * lr;
	   	auto b = b_input + gradient;
	    auto w = X * gradient;
		 
		auto w_vector = SGVector<float64_t>(num_feat);
		w_vector.set_const(NumericType{1} / num_feat);
		NumericType b_value{0};

		std::mt19937_64 prng(125);
		SGMatrix<NumericType> features = DataGenerator::generate_gaussians(
		    num_examples, 2, num_feat, prng);

		SGVector<float64_t> labels(num_examples);
		for (index_t i = 0; i < features.num_cols; ++i)
		{
			labels[i / 2] = (i < features.num_cols / 2) ? 1.0 : -1.0;
		}

		for (auto&& backend : this->m_backends)
		{
			if (backend != GRAPH_BACKEND::SHOGUN)
				continue;
			
			auto graph = make_shared<Graph>(vector{X, y, w_input, b_input, lr}, 
				vector<shared_ptr<node::Node>>{w, b});
			graph->build(backend);
			for (auto iter : range(max_iters))
			{
				for (const auto& idx : range(features.num_rows))
				{
					auto update = graph->evaluate(vector{make_shared<Tensor>(features.get_row_vector(idx)), 
						make_shared<Tensor>(labels[idx]),
						make_shared<Tensor>(w_vector),
						make_shared<Tensor>(b_value),
						make_shared<Tensor>(learning_rate)});
					w_vector = update[0]->template as<SGVector<NumericType>>();
					b_value = update[1]->template as<NumericType>();
				}
			}
			auto prediction_graph = make_shared<Graph>(vector{X, y, w_input, b_input}, 
				vector<shared_ptr<node::Node>>{correct});
			prediction_graph->build(backend);
			NumericType acc = 0;
			for (const auto& idx : range(features.num_rows))
			{
				auto result = prediction_graph->evaluate(vector{make_shared<Tensor>(features.get_row_vector(idx)), 
					make_shared<Tensor>(labels[idx]),
					make_shared<Tensor>(w_vector),
					make_shared<Tensor>(b_value)});
				acc += result[0]->template as<NumericType>();
			}
			std::cout << acc << '\n';
			std::cout << acc / features.num_rows << '\n';
		}	 
	}
#endif
}
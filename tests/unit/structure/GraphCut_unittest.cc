#include <shogun/structure/GraphCut.h>
#include <shogun/structure/FactorGraphDataGenerator.h>

#include <gtest/gtest.h>

using namespace shogun;

// Test max-flow algorithm on s-t graph
TEST(GraphCut, graph_cut_st_graph)
{
	int32_t num_nodes = 5;
	int32_t num_edges = 6;

	auto g = std::make_shared<GraphCut>(num_nodes, num_edges);


	g->add_tweights(0, 4, 0);
	g->add_tweights(1, 2, 0);
	g->add_tweights(2, 8, 0);
	g->add_tweights(2, 0, 4);
	g->add_tweights(3, 0, 7);
	g->add_tweights(4, 0, 5);

	g->add_edge(0, 2, 5, 0);
	g->add_edge(0, 3, 2, 0);
	g->add_edge(1, 2, 6, 0);
	g->add_edge(1, 4, 9, 0);
	g->add_edge(2, 3, 1, 0);
	g->add_edge(2, 4, 3, 0);

	g->init_maxflow();
	int32_t flow = g->compute_maxflow();
	EXPECT_EQ(flow, 12);

	std::vector<ETerminalType> expected_assignments;
	expected_assignments.push_back(SOURCE);
	expected_assignments.push_back(SOURCE);
	expected_assignments.push_back(SOURCE);
	expected_assignments.push_back(SINK);
	expected_assignments.push_back(SOURCE);

	for (int32_t i = 0; i < num_nodes; i++)
	{
		EXPECT_EQ(g->get_assignment(i), expected_assignments[i]);
	}


}

// Test graph-cuts inference for a simple two nodes chain structure graph
TEST(GraphCut, graph_cut_chain)
{
	auto fg_test_data = std::make_shared<FactorGraphDataGenerator>();


	auto fg = fg_test_data->simple_chain_graph();

	EXPECT_TRUE(fg->is_acyclic_graph());
	EXPECT_TRUE(fg->is_connected_graph());
	EXPECT_TRUE(fg->is_tree_graph());
	EXPECT_EQ(fg->get_num_edges(), 4);

	MAPInference infer_met(fg, GRAPH_CUT);
	infer_met.inference();

	auto fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();

	EXPECT_NEAR(0.4, infer_met.get_energy(), 1E-10);

}

// Test graph-cuts inference for a four nodes chain graph
// potentials are randomly generated
TEST(GraphCut, graph_cut_random)
{
	int32_t seed = 10;

	SGVector<int32_t> assignment_expected; // expected assignment
	float64_t min_energy_expected; // expected minimum energy

	auto fg_test_data = std::make_shared<FactorGraphDataGenerator>();
	fg_test_data->put("seed", seed);

	auto fg_random = fg_test_data->random_chain_graph(assignment_expected, min_energy_expected);

	MAPInference infer_met(fg_random, GRAPH_CUT);
	infer_met.inference();

	auto fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();

	EXPECT_EQ(assignment.size(), assignment_expected.size());

	for (int32_t i = 0; i < assignment.size(); i++)
		EXPECT_EQ(assignment[i], assignment_expected[i]);

	EXPECT_NEAR(min_energy_expected, infer_met.get_energy(), 1E-10);

}

// Test graph-cuts with SOSVM framework
// using randomly generated synthetic data
TEST(GraphCut, graph_cut_sosvm)
{
	auto fg_test_data = std::make_shared<FactorGraphDataGenerator>();


	EXPECT_NEAR(fg_test_data->test_sosvm(GRAPH_CUT), 0, 0.1);


}

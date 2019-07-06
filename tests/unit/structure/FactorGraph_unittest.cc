#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/Factor.h>
#include <gtest/gtest.h>

using namespace std;

using namespace shogun;

inline int grid_to_index(int32_t x, int32_t y, int32_t w = 10)
{
	return x + w*y;
}

inline void index_to_grid(int32_t index, int32_t& x, int32_t& y, int32_t w = 10)
{
	x = index % w;
	y = index / w;
}

TEST(FactorGraph, compute_energies_data_indep)
{
	// Create one simple pairwise factor type
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 1.0;
	w[1] = 0.2;
	w[2] = -0.2;
	w[3] = 1.0;
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	// Create a factor graph from the model: 3 binary variables
	SGVector<int32_t> vc(3);
	vc[0] = 2;
	vc[1] = 2;
	vc[2] = 2;
	FactorGraph fg(vc);

	// Add factors
	SGVector<float64_t> data;
	SGVector<int32_t> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	auto fac1 = std::make_shared<Factor>(factortype, var_index, data);

	fg.add_factor(fac1);

	var_index[0] = 1;
	var_index[1] = 2;
	auto fac2 = std::make_shared<Factor>(factortype, var_index, data);

	fg.add_factor(fac2);

	var_index[0] = 0;
	var_index[1] = 2;
	auto fac3 = std::make_shared<Factor>(factortype, var_index, data);

	fg.add_factor(fac3);

	fg.compute_energies();

	auto allfac = fg.get_factors();
	for (int32_t fi = 0; fi < 3; fi++)
	{
		auto ft = allfac[fi];
		SGVector<float64_t> energies = ft->get_energies();


		for (int32_t ei = 0; ei < energies.size(); ei++)
			EXPECT_NEAR(w[ei], energies[ei], 1E-10);
	}





}

TEST(FactorGraph, evaluate_energy_data_indep)
{
	// Create one simple pairwise factor type
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 0.0;
	w[1] = 0.3;
	w[2] = 0.2;
	w[3] = 0.0;
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	SGVector<int32_t> card1(1);
	card1[0] = 2;
	SGVector<float64_t> w1(2);
	w1[0] = 0.1;
	w1[1] = 0.7;
	int32_t tid1 = 1;
	auto factortype1a = std::make_shared<TableFactorType>(tid1, card1, w1);


	SGVector<float64_t> w2(2);
	w2[0] = 0.3;
	w2[1] = 0.6;
	int32_t tid2 = 2;
	auto factortype1b = std::make_shared<TableFactorType>(tid2, card1, w2);


	// Create a factor graph from the model: 2 binary variables
	SGVector<int32_t> vc(2);
	vc[0] = 2;
	vc[1] = 2;
	FactorGraph fg(vc);

	// Add factors
	SGVector<float64_t> data;
	SGVector<int32_t> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	auto fac1 = std::make_shared<Factor>(factortype, var_index, data);

	fg.add_factor(fac1);

	SGVector<int32_t> var_index1(1);
	var_index1[0] = 0;
	auto fac1a =std::make_shared< Factor>(factortype1a, var_index1, data);

	fg.add_factor(fac1a);

	SGVector<int32_t> var_index2(1);
	var_index2[0] = 1;
	auto fac1b =std::make_shared< Factor>(factortype1b, var_index2, data);

	fg.add_factor(fac1b);

	fg.compute_energies();

	SGVector<int32_t> state(2);
	state[0] = 0;
	state[1] = 0;
	EXPECT_NEAR(0.4, fg.evaluate_energy(state), 1E-10);

	state[0] = 0;
	state[1] = 1;
	EXPECT_NEAR(0.9, fg.evaluate_energy(state), 1E-10);

	state[0] = 1;
	state[1] = 0;
	EXPECT_NEAR(1.3, fg.evaluate_energy(state), 1E-10);

	state[0] = 1;
	state[1] = 1;
	EXPECT_NEAR(1.3, fg.evaluate_energy(state), 1E-10);







}

TEST(FactorGraph, evaluate_energy_data_dep)
{
	// Create one simple pairwise factor type
	SGVector<float64_t> w;
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	SGVector<int32_t> card1(1);
	card1[0] = 2;
	int32_t tid1 = 1;
	auto factortype1a = std::make_shared<TableFactorType>(tid1, card1, w);


	// Create a factor graph from the model: 2 binary variables
	SGVector<int32_t> vc(2);
	vc[0] = 2;
	vc[1] = 2;
	FactorGraph fg(vc);

	// Add factors
	SGVector<float64_t> data1(4);
	data1[0] = 0.0;
	data1[1] = 0.3;
	data1[2] = 0.2;
	data1[3] = 0.0;
	SGVector<int32_t> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	auto fac1 = std::make_shared<Factor>(factortype, var_index, data1);

	fg.add_factor(fac1);

	SGVector<float64_t> data2(2);
	data2[0] = 0.1;
	data2[1] = 0.7;
	SGVector<int32_t> var_index1(1);
	var_index1[0] = 0;
	auto fac1a =std::make_shared< Factor>(factortype1a, var_index1, data2);

	fg.add_factor(fac1a);

	SGVector<float64_t> data3(2);
	data3[0] = 0.3;
	data3[1] = 0.6;
	SGVector<int32_t> var_index2(1);
	var_index2[0] = 1;
	auto fac1b =std::make_shared< Factor>(factortype1a, var_index2, data3);

	fg.add_factor(fac1b);

	fg.compute_energies();

	// Evaluation
	SGVector<int32_t> state(2);
	state[0] = 0;
	state[1] = 0;
	EXPECT_NEAR(0.4, fg.evaluate_energy(state), 1E-10);

	state[0] = 0;
	state[1] = 1;
	EXPECT_NEAR(0.9, fg.evaluate_energy(state), 1E-10);

	state[0] = 1;
	state[1] = 0;
	EXPECT_NEAR(1.3, fg.evaluate_energy(state), 1E-10);

	state[0] = 1;
	state[1] = 1;
	EXPECT_NEAR(1.3, fg.evaluate_energy(state), 1E-10);






}

TEST(FactorGraph, evaluate_energy_param_data)
{
	// Create one simple pairwise factor type
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(8);
	w[0] = 0.3; // 0,0
	w[1] = 0.5;
	w[2] = 1.0; // 1,0
	w[3] = 0.2;
	w[4] = 0.05; // 0,1
	w[5] = 0.6;
	w[6] = -0.2; // 1,1
	w[7] = 0.75;
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	// Create a factor graph from the model: 3 binary variables
	SGVector<int32_t> vc(3);
	vc[0] = 2;
	vc[1] = 2;
	vc[2] = 2;
	FactorGraph fg(vc);

	// Add factors
	SGVector<float64_t> data(2);
	data[0] = 0.1;
	data[1] = 0.2;
	SGVector<int32_t> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	auto fac1 = std::make_shared<Factor>(factortype, var_index, data);

	fg.add_factor(fac1);

	SGVector<float64_t> data1(2);
	data1[0] = 0.3;
	data1[1] = 0.4;
	SGVector<int32_t> var_index1(2);
	var_index1[0] = 1;
	var_index1[1] = 2;
	auto fac1a =std::make_shared< Factor>(factortype, var_index1, data1);

	fg.add_factor(fac1a);

	SGVector<float64_t> data2(2);
	data2[0] = 0.5;
	data2[1] = 0.6;
	SGVector<int32_t> var_index2(2);
	var_index2[0] = 0;
	var_index2[1] = 2;
	auto fac1b =std::make_shared< Factor>(factortype, var_index2, data2);

	fg.add_factor(fac1b);

	fg.compute_energies();

	SGVector<float64_t> marginals(4);
	marginals[0] = 0.25;
	marginals[1] = 0.4;
	marginals[2] = 0.1;
	marginals[3] = 0.25;

	SGVector<float64_t> gradients(8);
	gradients.zero();
	auto allfac = fg.get_factors();
	//for (int32_t fi = 0; fi < allfac.size(); fi++)
	int32_t fi = 0;
	{
		auto ft = allfac[fi];
		ft->compute_gradients(marginals, gradients);

	}

	EXPECT_NEAR(0.025, gradients[0], 1E-10);
	EXPECT_NEAR(0.05, gradients[1], 1E-10);
	EXPECT_NEAR(0.04, gradients[2], 1E-10);
	EXPECT_NEAR(0.08, gradients[3], 1E-10);
	EXPECT_NEAR(0.01, gradients[4], 1E-10);
	EXPECT_NEAR(0.02, gradients[5], 1E-10);
	EXPECT_NEAR(0.025, gradients[6], 1E-10);
	EXPECT_NEAR(0.05, gradients[7], 1E-10);





}

TEST(FactorGraph, evaluate_energy_param_data_sparse)
{
	// Create one simple pairwise factor type
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(8);
	w[0] = 0.3; // 0,0
	w[1] = 0.5;
	w[2] = 1.0; // 1,0
	w[3] = 0.2;
	w[4] = 0.05; // 0,1
	w[5] = 0.6;
	w[6] = -0.2; // 1,1
	w[7] = 0.75;
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	// Create a factor graph from the model: 3 binary variables
	SGVector<int32_t> vc(3);
	vc[0] = 2;
	vc[1] = 2;
	vc[2] = 2;
	FactorGraph fg(vc);

	SGSparseVectorEntry<float64_t>* sdata = SG_MALLOC(SGSparseVectorEntry<float64_t>, 2);
	sdata[0].feat_index = 0;
	sdata[0].entry = 0.1;
	sdata[1].feat_index = 1;
	sdata[1].entry = 0.2;

	// Add factors
	SGSparseVector<float64_t> data(sdata, 2);
	SGVector<int32_t> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	auto fac1 = std::make_shared<Factor>(factortype, var_index, data);

	fg.add_factor(fac1);

	SGSparseVectorEntry<float64_t>* sdata1 = SG_MALLOC(SGSparseVectorEntry<float64_t>, 2);
	sdata1[0].feat_index = 0;
	sdata1[0].entry = 0.3;
	sdata1[1].feat_index = 1;
	sdata1[1].entry = 0.4;

	SGSparseVector<float64_t> data1(sdata1, 2);
	SGVector<int32_t> var_index1(2);
	var_index1[0] = 1;
	var_index1[1] = 2;
	auto fac1a =std::make_shared<Factor>(factortype, var_index1, data1);

	fg.add_factor(fac1a);

	SGSparseVectorEntry<float64_t>* sdata2 = SG_MALLOC(SGSparseVectorEntry<float64_t>, 2);
	sdata2[0].feat_index = 0;
	sdata2[0].entry = 0.5;
	sdata2[1].feat_index = 1;
	sdata2[1].entry = 0.6;

	SGSparseVector<float64_t> data2(sdata2, 2);
	SGVector<int32_t> var_index2(2);
	var_index2[0] = 0;
	var_index2[1] = 2;
	auto fac1b =std::make_shared<Factor>(factortype, var_index2, data2);

	fg.add_factor(fac1b);

	fg.compute_energies();

	SGVector<float64_t> marginals(4);
	marginals[0] = 0.25;
	marginals[1] = 0.4;
	marginals[2] = 0.1;
	marginals[3] = 0.25;

	SGVector<float64_t> gradients(8);
	gradients.zero();
	auto allfac = fg.get_factors();
	for (int32_t fi = 0; fi < allfac.size(); fi++)
	{
		auto ft = allfac[fi];
		ft->compute_gradients(marginals, gradients);
	}

	// factor 3
	EXPECT_NEAR(0.225, gradients[0], 1E-10);
	EXPECT_NEAR(0.3, gradients[1], 1E-10);
	EXPECT_NEAR(0.36, gradients[2], 1E-10);
	EXPECT_NEAR(0.48, gradients[3], 1E-10);
	EXPECT_NEAR(0.09, gradients[4], 1E-10);
	EXPECT_NEAR(0.12, gradients[5], 1E-10);
	EXPECT_NEAR(0.225, gradients[6], 1E-10);
	EXPECT_NEAR(0.3, gradients[7], 1E-10);





}

TEST(FactorGraph, structure_analysis)
{
	int hh = 3;
	int ww = 3;

	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 0.0; // 0,0
	w[1] = 0.5; // 1,0
	w[2] = 0.5; // 0,1
	w[3] = 0.0; // 1,1
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	SGVector<int32_t> vc(hh*ww);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	FactorGraph fg(vc);

	// Add factors
	for (int32_t x = 0; x < ww; x++)
	{
		for (int32_t y = 0; y < hh; y++)
		{
			if (x > 0)
			{
				SGVector<float64_t> data;
				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x,y,ww);
				var_index[1] = grid_to_index(x-1,y,ww);
				auto fac1 = std::make_shared<Factor>(factortype, var_index, data);
				fg.add_factor(fac1);
			}

			if (x == 0 && y > 0)
			{
				SGVector<float64_t> data;
				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x,y-1,ww);
				var_index[1] = grid_to_index(x,y,ww);
				auto fac1 = std::make_shared<Factor>(factortype, var_index, data);
				fg.add_factor(fac1);
			}
		}
	}


	fg.connect_components();

	EXPECT_TRUE(fg.is_acyclic_graph());
	EXPECT_TRUE(fg.is_connected_graph());
	EXPECT_TRUE(fg.is_tree_graph());
	EXPECT_EQ(fg.get_num_edges(), 16);
}

TEST(FactorGraph, structure_analysis_loopy)
{
	int hh = 3;
	int ww = 3;

	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 0.0; // 0,0
	w[1] = 0.5; // 1,0
	w[2] = 0.5; // 0,1
	w[3] = 0.0; // 1,1
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	SGVector<int32_t> vc(hh*ww);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	FactorGraph fg(vc);

	// Add factors
	for (int32_t x = 0; x < ww; x++)
	{
		for (int32_t y = 0; y < hh; y++)
		{
			if (x > 0)
			{
				SGVector<float64_t> data;
				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x,y,ww);
				var_index[1] = grid_to_index(x-1,y,ww);
				auto fac1 = std::make_shared<Factor>(factortype, var_index, data);
				fg.add_factor(fac1);
			}

			if (y > 0)
			{
				SGVector<float64_t> data;
				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x,y-1,ww);
				var_index[1] = grid_to_index(x,y,ww);
				auto fac1 = std::make_shared<Factor>(factortype, var_index, data);
				fg.add_factor(fac1);
			}
		}
	}


	fg.connect_components();

	EXPECT_FALSE(fg.is_acyclic_graph());
	EXPECT_TRUE(fg.is_connected_graph());
	EXPECT_FALSE(fg.is_tree_graph());
	EXPECT_EQ(fg.get_num_edges(), 24);
}

TEST(FactorGraph, structure_analysis_disconnected)
{
	int hh = 3;
	int ww = 3;

	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 0.0; // 0,0
	w[1] = 0.5; // 1,0
	w[2] = 0.5; // 0,1
	w[3] = 0.0; // 1,1
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	SGVector<int32_t> vc(hh*ww);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	FactorGraph fg(vc);

	// Add factors
	for (int32_t x = 0; x < ww; x++)
	{
		for (int32_t y = 0; y < hh; y++)
		{
			if (x > 0)
			{
				SGVector<float64_t> data;
				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x,y,ww);
				var_index[1] = grid_to_index(x-1,y,ww);
				auto fac1 = std::make_shared<Factor>(factortype, var_index, data);
				fg.add_factor(fac1);
			}
		}
	}


	fg.connect_components();

	EXPECT_TRUE(fg.is_acyclic_graph());
	EXPECT_FALSE(fg.is_connected_graph());
	EXPECT_FALSE(fg.is_tree_graph());
	EXPECT_EQ(fg.get_num_edges(), 12);
}

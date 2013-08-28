#include <shogun/io/SGIO.h>
#include <shogun/base/init.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/Factor.h>
#include <shogun/labels/FactorGraphLabels.h>
#include <shogun/structure/MAPInference.h>
#include <gtest/gtest.h>

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

TEST(BeliefPropagation, tree_max_product_string)
{
	// ftype
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 0.0; // 0,0
	w[1] = 0.3; // 1,0
	w[2] = 0.2; // 0,1
	w[3] = 0.0; // 1,1
	SGString<char> tid((char*)("pairwise"), 8);
	CTableFactorType* factortype = new CTableFactorType(tid, card, w);
	SG_REF(factortype);

	SGVector<int32_t> card1(1);
	card1[0] = 2;
	SGVector<float64_t> w1(2);
	w1[0] = 0.1;
	w1[1] = 0.7;
	SGString<char> tid1((char*)("unary1"), 6);
	CTableFactorType* factortype1 = new CTableFactorType(tid1, card1, w1);
	SG_REF(factortype1);

	SGVector<int32_t> card2(1);
	card2[0] = 2;
	SGVector<float64_t> w2(2);
	w2[0] = 0.3;
	w2[1] = 0.6;
	SGString<char> tid2((char*)("unary2"), 6);
	CTableFactorType* factortype2 = new CTableFactorType(tid2, card2, w2);
	SG_REF(factortype2);

	// fg
	SGVector<int32_t> vc(2);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	CFactorGraph* fg = new CFactorGraph(vc);
	SG_REF(fg);

	// add factors
	SGVector<float64_t> data;
	SGVector<int32_t> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	CFactor* fac1 = new CFactor(factortype, var_index, data);
	fg->add_factor(fac1);

	SGVector<int32_t> var_index1(1);
	var_index1[0] = 0;
	CFactor* fac2 = new CFactor(factortype1, var_index1, data);
	fg->add_factor(fac2);

	SGVector<int32_t> var_index2(1);
	var_index2[0] = 1;
	CFactor* fac3 = new CFactor(factortype2, var_index2, data);
	fg->add_factor(fac3);

	SG_UNREF(factortype);
	SG_UNREF(factortype1);
	SG_UNREF(factortype2);

	// energy table
	fg->compute_energies();

	fg->connect_components();

	EXPECT_TRUE(fg->is_acyclic_graph());
	EXPECT_TRUE(fg->is_connected_graph());
	EXPECT_TRUE(fg->is_tree_graph());
	EXPECT_EQ(fg->get_num_edges(), 4);

	CMAPInference infer_met(fg, TREE_MAX_PROD);	
	infer_met.inference();

	CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();
	SG_UNREF(fg_observ);

	EXPECT_NEAR(0.4, infer_met.get_energy(), 1E-10);

	SG_UNREF(fg);
}

TEST(BeliefPropagation, tree_max_product_random)
{
	CMath::init_random(17);
	int N = 2;
	for (int32_t rani = 0; rani < 10; rani++)
	{
		// ftype
		SGVector<int32_t> card(2);
		card[0] = 2;
		card[1] = 2;
		SGVector<float64_t> w;
		SGString<char> tid((char*)("pairwise"), 8);
		CTableFactorType* factortype = new CTableFactorType(tid, card, w);
		SG_REF(factortype);

		SGVector<int32_t> card1(1);
		card1[0] = 2;
		SGVector<float64_t> w1;
		SGString<char> tid1((char*)("unary1"), 6);
		CTableFactorType* factortype1 = new CTableFactorType(tid1, card1, w1);
		SG_REF(factortype1);

		// fg
		SGVector<int32_t> vc(N*N);
		SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
		CFactorGraph* fg = new CFactorGraph(vc);
		SG_REF(fg);

		// Add factors
		for (int32_t y = 0; y < N; ++y)
			for (int32_t x = 0; x < N; ++x)
			{
				SGVector<float64_t> data(2);
				data[0] = CMath::random(0.0, 1.0);
				data[1] = CMath::random(0.0, 1.0);

				SGVector<int32_t> var_index(1);
				var_index[0] = y*N + x;

				CFactor* fac1 = new CFactor(factortype1, var_index, data);
				fg->add_factor(fac1);
			}

		for (int32_t x = 0; x < N; x++)
		{
			for (int32_t y = 0; y < N; y++)
			{
				if (x > 0)
				{
					SGVector<float64_t> data(4);
					data[0] = CMath::random(0.0, 1.0);
					data[1] = CMath::random(0.0, 1.0);
					data[2] = CMath::random(0.0, 1.0);
					data[3] = CMath::random(0.0, 1.0);
	
					SGVector<int32_t> var_index(2);
					var_index[0] = grid_to_index(x,y,N);
					var_index[1] = grid_to_index(x-1,y,N);
					CFactor* fac1 = new CFactor(factortype, var_index, data);
					fg->add_factor(fac1);
				}

				if (x == 0 && y > 0)
				{
					SGVector<float64_t> data(4);
					data[0] = CMath::random(0.0, 1.0);
					data[1] = CMath::random(0.0, 1.0);
					data[2] = CMath::random(0.0, 1.0);
					data[3] = CMath::random(0.0, 1.0);
	
					SGVector<int32_t> var_index(2);
					var_index[0] = grid_to_index(x,y-1,N);
					var_index[1] = grid_to_index(x,y,N);
					CFactor* fac1 = new CFactor(factortype, var_index, data);
					fg->add_factor(fac1);
				}
			}
		}

		SG_UNREF(factortype);
		SG_UNREF(factortype1);

		// energy table
		fg->compute_energies();

		fg->connect_components();

		EXPECT_TRUE(fg->is_acyclic_graph());
		EXPECT_TRUE(fg->is_connected_graph());
		EXPECT_TRUE(fg->is_tree_graph());
		EXPECT_EQ(fg->get_num_edges(), 10);

		CMAPInference infer_met(fg, TREE_MAX_PROD);	
		infer_met.inference();

		CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();
		SGVector<int32_t> assignment = fg_observ->get_data();
		SG_UNREF(fg_observ);

		// Find minimum energy state by exhaustive search
		SGVector<int> test_var(N*N);
		SGVector<int> min_var(N*N);
		double min_var_energy = std::numeric_limits<double>::infinity();
		for (int v0 = 0; v0 < 2; ++v0) {
			test_var[0] = v0;
			for (int v1 = 0; v1 < 2; ++v1) {
				test_var[1] = v1;
				for (int v2 = 0; v2 < 2; ++v2) {
					test_var[2] = v2;
					for (int v3 = 0; v3 < 2; ++v3) {
						test_var[3] = v3;

						double orig_e = fg->evaluate_energy(test_var);
						if (orig_e < min_var_energy) {
							min_var = test_var.clone();
							min_var_energy = orig_e;
						}
					}
				}
			}
		}

		for (int32_t si = 0; si < N*N; si++)
			EXPECT_EQ(assignment[si], min_var[si]);

		EXPECT_NEAR(infer_met.get_energy(), min_var_energy, 1E-10);

		SG_UNREF(fg);
	}
}

#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/Factor.h>
#include <shogun/labels/FactorGraphLabels.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/FGTestData.h>

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

float64_t hamming_loss(SGVector<int32_t> y_truth, SGVector<int32_t> y_pred)
{
	float64_t loss = 0.0;
	for (int32_t i = 0; i < y_truth.size(); i++)
	{
		if (y_truth[i] != y_pred[i])
			loss += 1;
	}
	return (loss / y_truth.vlen);
}

TEST(BeliefPropagation, tree_max_product_string)
{
	CFGTestData* fg_test_data = new CFGTestData();
	SG_REF(fg_test_data);

	CFactorGraph* fg = fg_test_data->simple_chain_graph();

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
	SG_UNREF(fg_test_data);
}

TEST(BeliefPropagation, tree_max_product_random)
{
	SGVector<int32_t> assignment_expected; // expected assignment
	float64_t min_energy_expected; // expected minimum energy

	CFGTestData* fg_test_data = new CFGTestData();
	SG_REF(fg_test_data);
	CFactorGraph* fg = fg_test_data->random_chain_graph(assignment_expected, min_energy_expected);
	
	EXPECT_TRUE(fg->is_acyclic_graph());
	EXPECT_TRUE(fg->is_connected_graph());
	EXPECT_TRUE(fg->is_tree_graph());
	EXPECT_EQ(fg->get_num_edges(), 10);

	CMAPInference infer_met(fg, TREE_MAX_PROD);
	infer_met.inference();

	CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();
	SG_UNREF(fg_observ);
	
	EXPECT_EQ(assignment.size(), assignment_expected.size());

	for (int32_t i = 0; i < assignment.size(); i++)
		EXPECT_EQ(assignment[i], assignment_expected[i]);

	EXPECT_NEAR(min_energy_expected, infer_met.get_energy(), 1E-10);

	SG_UNREF(fg_test_data);
	SG_UNREF(fg);
}

TEST(BeliefPropagation, loss_augmented_energies)
{
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 0.0; // 0,0
	w[1] = 0.0; // 1,0
	w[2] = 0.0; // 0,1
	w[3] = 0.0; // 1,1
	int32_t tid = 0;
	CTableFactorType* factortype = new CTableFactorType(tid, card, w);
	SG_REF(factortype);

	SGVector<int32_t> vc(3);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	CFactorGraph* fg = new CFactorGraph(vc);
	SG_REF(fg);

	SGVector<float64_t> data(1);
	data[0] = 1.0;
	SGVector<int32_t> var_index1(2);
	var_index1[0] = 0;
	var_index1[1] = 1;
	CFactor* fac1 = new CFactor(factortype, var_index1, data);
	fg->add_factor(fac1);

	SGVector<int32_t> var_index2(2);
	var_index2[0] = 1;
	var_index2[1] = 2;
	CFactor* fac2 = new CFactor(factortype, var_index2, data);
	fg->add_factor(fac2);

	fg->connect_components();
	fg->compute_energies();

	SGVector<int32_t> y_truth(3);
	y_truth.zero();
	fg->loss_augmentation(y_truth);

	SGVector<int32_t> y_cand(3);
	for (int i = 0; i < 2; i++)
	{
		y_cand[0] = i;
		for (int j = 0; j < 2; j++)
		{
			y_cand[1] = j;
			for (int k = 0; k < 2; k++)
			{
				y_cand[2] = k;
				float64_t loss_eg = fg->evaluate_energy(y_cand);
				EXPECT_NEAR(0-hamming_loss(y_truth,y_cand), loss_eg, 1E-10);
			}
		}
	}

	SG_UNREF(fg);
	SG_UNREF(factortype);
}

TEST(BeliefPropagation, tree_max_product_multi_states)
{
	CFGTestData* fg_test_data = new CFGTestData();
	SG_REF(fg_test_data);

	CFactorGraph* fg = fg_test_data->multi_state_tree_graph();

	CMAPInference infer_met(fg, TREE_MAX_PROD);
	infer_met.inference();

	CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();
	EXPECT_EQ(assignment[0],2);
	EXPECT_EQ(assignment[1],0);
	EXPECT_EQ(assignment[2],2);

	EXPECT_NEAR(-3.8, infer_met.get_energy(), 1E-10);
	EXPECT_NEAR(-3.8, fg->evaluate_energy(assignment), 1E-10);

	SG_UNREF(fg_observ);
	SG_UNREF(fg);
	SG_UNREF(fg_test_data);
}


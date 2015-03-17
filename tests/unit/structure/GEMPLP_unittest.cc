#include <shogun/structure/GEMPLP.h>
#include <shogun/structure/FactorGraphDataGenerator.h>
#include <gtest/gtest.h>

using namespace shogun;

// Test find intersection index
TEST(GEMPLP, find_intersections_index)
{
	CGEMPLP* mplp = new CGEMPLP();
	SG_REF(mplp);
	
	SGVector<int32_t> clique_A(3);
	SGVector<int32_t> clique_B(3);
	
	for (int32_t i = 0; i < 3; i++)
	{
		clique_A[i] = i;
		clique_B[i] = i + 3;
	}

	int32_t k;
	k = mplp->find_intersection_index(clique_A, clique_B);
	EXPECT_EQ(k, -1);

	clique_B[0] = 2;
	k = mplp->find_intersection_index(clique_A, clique_B);
	EXPECT_EQ(k, 0);
	EXPECT_EQ(mplp->m_all_intersections.size(), 1);
	EXPECT_EQ(mplp->m_all_intersections[0][0], 2);	

	clique_B[1] = 1;
	k = mplp->find_intersection_index(clique_A, clique_B);
	EXPECT_EQ(k, 1);
	EXPECT_EQ(mplp->m_all_intersections.size(), 2);
	EXPECT_EQ(mplp->m_all_intersections[1].size(), 2);	
	EXPECT_EQ(mplp->m_all_intersections[1][0], 1);	
	EXPECT_EQ(mplp->m_all_intersections[1][1], 2);	
	
	SG_UNREF(mplp);
}

// Test find maximum value in sub-array
TEST(GEMPLP, max_in_subdimension)
{
	CGEMPLP* mplp = new CGEMPLP();
	SG_REF(mplp);

	// dimensions of the target array 2x2
	SGVector<int32_t> dims_tar(2);
	dims_tar[0] = 2;
	dims_tar[1] = 2;
	// dimensions of the result (sub) array 1x2
	SGVector<int32_t> dims_max(1);
	dims_max[0] = 2;

	// define target nd array
	SGNDArray<float64_t> arr_tar(dims_tar);	
	arr_tar[0] = 0; // (0,0)	
	arr_tar[1] = 1;	// (0,1)
	arr_tar[2] = 2;	// (1,0)
	arr_tar[3] = 3;	// (1,1)

	// define result nd array
	SGNDArray<float64_t> arr_max(dims_max);
	
	// define sub-dimension	
	SGVector<int32_t> subset_inds(1);
	subset_inds[0] = 0;

	// perform max operation
	mplp->max_in_subdimension(arr_tar, subset_inds, arr_max);
	EXPECT_EQ(arr_max[0],1);
	EXPECT_EQ(arr_max[1],3);

	subset_inds[0] = 1;
	mplp->max_in_subdimension(arr_tar, subset_inds, arr_max);
	EXPECT_EQ(arr_max[0],2);
	EXPECT_EQ(arr_max[1],3);

	SG_UNREF(mplp);
}

// Test initialization
TEST(GEMPLP, initialization)
{
	CFactorGraphDataGenerator* fg_test_data = new CFactorGraphDataGenerator();
	SG_REF(fg_test_data);

	CFactorGraph* fg = fg_test_data->simple_chain_graph();
	CGEMPLP* mplp = new CGEMPLP(fg);
	SG_REF(mplp);

	EXPECT_EQ(mplp->m_all_intersections[0][0],0);
	EXPECT_EQ(mplp->m_all_intersections[0][1],1);
	EXPECT_EQ(mplp->m_all_intersections[1][0],0);
	EXPECT_EQ(mplp->m_all_intersections[2][0],1);

	SG_UNREF(fg_test_data);
	SG_UNREF(fg);
	SG_UNREF(mplp);
}

// Test convert message
TEST(GEMPLP, convert_energy_to_potential)
{
	CFactorGraphDataGenerator* fg_test_data = new CFactorGraphDataGenerator();
	SG_REF(fg_test_data);

	CFactorGraph* fg = fg_test_data->simple_chain_graph();
	CGEMPLP* mplp = new CGEMPLP(fg);
	SG_REF(mplp);

	CFactor* factor = dynamic_cast<CFactor*>(mplp->m_factors->get_element(0));
	
	SGNDArray<float64_t> message = mplp->convert_energy_to_potential(factor);
	
	EXPECT_EQ(message.len_array, 4);
	EXPECT_EQ(message.array[0], -0.0);
	EXPECT_EQ(message.array[1], -0.2);
	EXPECT_EQ(message.array[2], -0.3);
	EXPECT_EQ(message.array[3], -0.0);

	SG_UNREF(fg_test_data);
	SG_UNREF(factor);
	SG_UNREF(fg);
	SG_UNREF(mplp);
}

// Test inference on simple chain graph
TEST(GEMPLP, simple_chain)
{
	CFactorGraphDataGenerator* fg_test_data = new CFactorGraphDataGenerator();
	SG_REF(fg_test_data);

	CFactorGraph* fg_simple = fg_test_data->simple_chain_graph();
	
	CMAPInference infer_met(fg_simple, GEMPLP);
	infer_met.inference();

	CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();
	SG_UNREF(fg_observ);
	
	EXPECT_EQ(assignment[0],0);
	EXPECT_EQ(assignment[1],0);
	EXPECT_NEAR(0.4, infer_met.get_energy(), 1E-10);

	SG_UNREF(fg_simple);
	SG_UNREF(fg_test_data);
}

// Test inference on random chain graph
TEST(GEMPLP, random_chain)
{
	CFactorGraphDataGenerator* fg_test_data = new CFactorGraphDataGenerator();
	SG_REF(fg_test_data);

	SGVector<int32_t> assignment_expected; // expected assignment
	float64_t min_energy_expected; // expected minimum energy

	CFactorGraph* fg_random = fg_test_data->random_chain_graph(assignment_expected, min_energy_expected);
		
	CMAPInference infer_met(fg_random, GEMPLP);
	infer_met.inference();

	CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();
	SG_UNREF(fg_observ);
	
	EXPECT_EQ(assignment.size(), assignment_expected.size());

	for (int32_t i = 0; i < assignment.size(); i++)
		EXPECT_EQ(assignment[i], assignment_expected[i]);

	EXPECT_NEAR(min_energy_expected, infer_met.get_energy(), 1E-10);

	SG_UNREF(fg_random);
	SG_UNREF(fg_test_data);
}

// Test with SOSVM
TEST(GEMPLP, sosvm)
{
	CFactorGraphDataGenerator* fg_test_data = new CFactorGraphDataGenerator();
	SG_REF(fg_test_data);

	EXPECT_EQ(fg_test_data->test_sosvm(GEMPLP), 0);

	SG_UNREF(fg_test_data);
}

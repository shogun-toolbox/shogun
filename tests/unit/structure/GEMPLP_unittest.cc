#include <shogun/structure/GEMPLP.h>
#include <shogun/structure/FactorGraphDataGenerator.h>
#include <gtest/gtest.h>

using namespace shogun;

// Test find intersection index
TEST(GEMPLP, find_intersections_index)
{
	auto mplp = std::make_shared<GEMPLP>();
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
	EXPECT_EQ(mplp->m_all_intersections.size(), 1u);
	EXPECT_EQ(mplp->m_all_intersections[0][0], 2);

	clique_B[1] = 1;
	k = mplp->find_intersection_index(clique_A, clique_B);
	EXPECT_EQ(k, 1);
	EXPECT_EQ(mplp->m_all_intersections.size(), 2u);
	EXPECT_EQ(mplp->m_all_intersections[1].size(), 2);
	EXPECT_EQ(mplp->m_all_intersections[1][0], 1);
	EXPECT_EQ(mplp->m_all_intersections[1][1], 2);
}

// Test find maximum value in sub-array
TEST(GEMPLP, max_in_subdimension)
{
	auto mplp = std::make_shared<GEMPLP>();
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
}

// Test initialization
TEST(GEMPLP, initialization)
{
	auto fg_test_data = std::make_shared<FactorGraphDataGenerator>();


	auto fg = fg_test_data->simple_chain_graph();
	auto mplp = std::make_shared<GEMPLP>(fg);


	EXPECT_EQ(mplp->m_all_intersections[0][0],0);
	EXPECT_EQ(mplp->m_all_intersections[0][1],1);
	EXPECT_EQ(mplp->m_all_intersections[1][0],0);
	EXPECT_EQ(mplp->m_all_intersections[2][0],1);

}

// Test convert message
TEST(GEMPLP, convert_energy_to_potential)
{
	auto fg_test_data = std::make_shared<FactorGraphDataGenerator>();


	auto fg = fg_test_data->simple_chain_graph();
	auto mplp = std::make_shared<GEMPLP>(fg);


	auto factor = mplp->m_factors[0];

	SGNDArray<float64_t> message = mplp->convert_energy_to_potential(factor);

	EXPECT_EQ(message.len_array, 4);
	EXPECT_EQ(message.array[0], -0.0);
	EXPECT_EQ(message.array[1], -0.2);
	EXPECT_EQ(message.array[2], -0.3);
	EXPECT_EQ(message.array[3], -0.0);


}

// Test inference on simple chain graph
TEST(GEMPLP, simple_chain)
{
	auto fg_test_data = std::make_shared<FactorGraphDataGenerator>();


	auto fg_simple = fg_test_data->simple_chain_graph();

	MAPInference infer_met(fg_simple, GEMP_LP);
	infer_met.inference();

	auto fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();

	EXPECT_EQ(assignment[0],0);
	EXPECT_EQ(assignment[1],0);
	EXPECT_NEAR(0.4, infer_met.get_energy(), 1E-10);

}

// Test inference on random chain graph
TEST(GEMPLP, random_chain)
{
	auto fg_test_data = std::make_shared<FactorGraphDataGenerator>();


	SGVector<int32_t> assignment_expected; // expected assignment
	float64_t min_energy_expected; // expected minimum energy

	auto fg_random = fg_test_data->random_chain_graph(assignment_expected, min_energy_expected);

	MAPInference infer_met(fg_random, GEMP_LP);
	infer_met.inference();

	auto fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();

	EXPECT_EQ(assignment.size(), assignment_expected.size());

	for (int32_t i = 0; i < assignment.size(); i++)
		EXPECT_EQ(assignment[i], assignment_expected[i]);

	EXPECT_NEAR(min_energy_expected, infer_met.get_energy(), 1E-10);

}

// Test with SOSVM
TEST(GEMPLP, sosvm)
{
	auto fg_test_data = std::make_shared<FactorGraphDataGenerator>();


	EXPECT_EQ(fg_test_data->test_sosvm(GEMP_LP), 0);


}

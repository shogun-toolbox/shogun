#include <shogun/ensemble/WeightedMajorityVote.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/labels/Labels.h>
#include <gtest/gtest.h>

using namespace shogun;

void generate_random_ensemble_matrix(SGMatrix<float64_t>& em,
	SGVector<float64_t>& expected_cv,
	const SGVector<float64_t>& w)
{
	int32_t num_classes = 3;
	for (index_t i = 0; i < em.num_rows; ++i)
	{
		SGVector<float64_t> hist(num_classes);
		hist.zero();
		float64_t max = CMath::ALMOST_NEG_INFTY;
		for (index_t j = 0; j < em.num_cols; ++j)
		{
			int32_t r = sg_rand->random(0, num_classes-1);
			em(i,j) = r;
			hist[r] += w[j];
			// if there's a tie mark it the first element will be the winner
			// i.e. whoever reached the max the first time.
			// SGVector<float64>::arg_max does not follow this logic...
			if (max < hist[r])
			{
				expected_cv[i] = r;
				max = hist[r];
			}
		}
	}
}

TEST(WeightedMajorityVote, combine_matrix)
{
	int32_t num_vectors = 20;
	int32_t num_classifiers = 5;
	SGMatrix<float64_t> ensemble_matrix(num_vectors, num_classifiers);
	SGVector<float64_t> expected(num_vectors);
	SGVector<float64_t> weights(num_classifiers);
	weights.random(0.5, 2.0);
	CWeightedMajorityVote* mv = new CWeightedMajorityVote(weights);

	expected.zero();

	generate_random_ensemble_matrix(ensemble_matrix, expected, weights);
	SGVector<float64_t> cv = mv->combine(ensemble_matrix);
	EXPECT_EQ(num_vectors, cv.vlen);

	for (index_t i = 0; i < cv.vlen; ++i)
		EXPECT_DOUBLE_EQ(expected[i], cv[i]);

	SG_UNREF(mv);
}

TEST(WeightedMajorityVote, binary_combine_vector)
{
	int32_t num_classifiers = 50;
	SGVector<float64_t> weights(num_classifiers);
	weights.random(0.5, 2.0);
	CWeightedMajorityVote* mv = new CWeightedMajorityVote(weights);
	SGVector<float64_t> v(num_classifiers);
	SGVector<float64_t> expected(2);

	float64_t max = 0;
	int64_t max_label = -10;

	expected.zero();
	v.zero();

	for (index_t i = 0; i < num_classifiers; ++i)
	{
		int32_t r = sg_rand->random(0, 1);
		v[i] = (r == 0) ? -1 : r;

		expected[r] += weights[i];
		if (max < expected[r])
		{
			max = expected[r];
			max_label = (r == 0) ? -1 : r;
		}
	}

	float64_t combined_label = mv->combine(v);
	EXPECT_EQ(max_label, combined_label);

	SG_UNREF(mv);
}

TEST(WeightedMajorityVote, multiclass_combine_vector)
{
	int32_t num_classifiers = 10;
	SGVector<float64_t> weights(num_classifiers);
	weights.random(0.5, 2.0);
	CWeightedMajorityVote* mv = new CWeightedMajorityVote(weights);
	SGVector<float64_t> v(num_classifiers);
	SGVector<float64_t> hist(3);

	hist.zero();
	v.zero();

	int64_t max_label = -1;
	float64_t max = -1;
	for (index_t i = 0; i < num_classifiers; ++i)
	{
		v[i] = sg_rand->random(0, 2);
		hist[v[i]] += weights[i];
		if (max < hist[v[i]])
		{
			max = hist[v[i]];
			max_label = v[i];
		}
	}
	float64_t c = mv->combine(v);

	EXPECT_EQ(max_label, c);

	SG_UNREF(mv);
}

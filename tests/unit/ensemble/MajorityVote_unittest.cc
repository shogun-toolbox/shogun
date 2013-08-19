#include <shogun/ensemble/MajorityVote.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/labels/Labels.h>
#include <gtest/gtest.h>

using namespace shogun;

extern void generate_random_ensemble_matrix(SGMatrix<float64_t>& em, 
	SGVector<float64_t>& expected_cv, 
	const SGVector<float64_t>& w);

TEST(MajorityVote, combine_matrix)
{
	int32_t num_vectors = 20;
	int32_t num_classifiers = 5;
	SGMatrix<float64_t> ensemble_matrix(num_vectors, num_classifiers);
	SGVector<float64_t> expected(num_vectors);
	SGVector<float64_t> w(num_classifiers);
	CMajorityVote* mv = new CMajorityVote();

	expected.zero();
	w.set_const(1.0);

	generate_random_ensemble_matrix(ensemble_matrix, expected, w);
	SGVector<float64_t> cv = mv->combine(ensemble_matrix);

	EXPECT_EQ(num_vectors, cv.vlen);

	for (index_t i = 0; i < cv.vlen; ++i)
		EXPECT_DOUBLE_EQ(expected[i], cv[i]);

	SG_UNREF(mv);
}

TEST(MajorityVote, binary_combine_vector)
{
	int32_t num_classifiers = 50;
	CMajorityVote* mv = new CMajorityVote();
	SGVector<float64_t> v(num_classifiers);
	SGVector<index_t> expected(2);
	int64_t max = 0;
	int64_t max_label = -10;

	expected.zero();
	v.zero();
	
	for (index_t i = 0; i < num_classifiers; ++i)
	{
		int32_t r = sg_rand->random(0, 1);
		v[i] = (r == 0) ? -1 : r;

		if (max < ++expected[r])
		{
			max = expected[r];
			max_label = (r == 0) ? -1 : r;
		}
	}

	float64_t combined_label = mv->combine(v);
	EXPECT_EQ(max_label, combined_label);

	SG_UNREF(mv);	
}

TEST(MajorityVote, multiclass_combine_vector)
{
	int32_t num_classifiers = 10;
	CMajorityVote* mv = new CMajorityVote();
	SGVector<float64_t> v(num_classifiers);
	SGVector<index_t> hist(3);

	v.zero();
	hist.zero();

	int64_t max_label = -1;
	int64_t max = -1;
	for (index_t i = 0; i < num_classifiers; ++i)
	{
		v[i] = sg_rand->random(0, 2);
		if (max < ++hist[v[i]])
		{
			max = hist[v[i]];
			max_label = v[i];
		}
	}

	float64_t c = mv->combine(v);

	EXPECT_EQ(max_label, c);

	SG_UNREF(mv);	
}

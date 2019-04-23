#include <shogun/ensemble/MajorityVote.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <gtest/gtest.h>

#include <random>

using namespace shogun;

extern void generate_random_ensemble_matrix(SGMatrix<float64_t>& em,
	SGVector<float64_t>& expected_cv,
	const SGVector<float64_t>& w, std::mt19937_64&);

TEST(MajorityVote, combine_matrix)
{
	std::mt19937_64 prng(32);

	int32_t num_vectors = 20;
	int32_t num_classifiers = 5;
	SGMatrix<float64_t> ensemble_matrix(num_vectors, num_classifiers);
	SGVector<float64_t> expected(num_vectors);
	SGVector<float64_t> w(num_classifiers);
	auto mv = std::make_shared<MajorityVote>();

	expected.zero();
	w.set_const(1.0);

	generate_random_ensemble_matrix(ensemble_matrix, expected, w, prng);
	SGVector<float64_t> cv = mv->combine(ensemble_matrix);

	EXPECT_EQ(num_vectors, cv.vlen);

	for (index_t i = 0; i < cv.vlen; ++i)
		EXPECT_DOUBLE_EQ(expected[i], cv[i]);

	
}

TEST(MajorityVote, binary_combine_vector)
{
	std::mt19937_64 prng(2);

	int32_t num_classifiers = 50;
	auto mv = std::make_shared<MajorityVote>();
	SGVector<float64_t> v(num_classifiers);
	SGVector<index_t> expected(2);
	int64_t max = 0;
	int64_t max_label = -10;

	expected.zero();
	v.zero();

	UniformIntDistribution<int32_t> uniform_int_dist(0, 1);
	for (index_t i = 0; i < num_classifiers; ++i)
	{
		int32_t r = uniform_int_dist(prng);
		v[i] = (r == 0) ? -1 : r;

		if (max < ++expected[r])
		{
			max = expected[r];
			max_label = (r == 0) ? -1 : r;
		}
	}

	float64_t combined_label = mv->combine(v);
	EXPECT_EQ(max_label, combined_label);

	
}

TEST(MajorityVote, multiclass_combine_vector)
{
	std::mt19937_64 prng(2);

	int32_t num_classifiers = 10;
	auto mv = std::make_shared<MajorityVote>();
	SGVector<float64_t> v(num_classifiers);
	SGVector<index_t> hist(3);

	v.zero();
	hist.zero();

	int64_t max_label = -1;
	int64_t max = -1;
	UniformIntDistribution<int32_t> uniform_int_dist(0, 2);
	for (index_t i = 0; i < num_classifiers; ++i)
	{
		v[i] = uniform_int_dist(prng);
		if (max < ++hist[index_t(v[i])])
		{
			max = hist[index_t(v[i])];
			max_label = v[i];
		}
	}

	float64_t c = mv->combine(v);

	EXPECT_EQ(max_label, c);

	
}

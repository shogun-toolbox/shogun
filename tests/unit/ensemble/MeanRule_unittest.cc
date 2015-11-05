#include <shogun/ensemble/MeanRule.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/linalg.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace linalg;


void generate_random_ensemble_matrix(SGMatrix<float64_t>& em)
{
	/* generate random ensemble classification matrix */
	for (index_t i = 0; i < em.num_cols; ++i)
	{
		float64_t* v = em.get_column_vector(i);
		SGVector<float64_t>::random_vector(v, em.num_rows, 0.0, 50.0);
	}
}

TEST(MeanRule, combine_matrix)
{
	int32_t num_vectors = 20;
	int32_t num_classifiers = 5;
	CMeanRule* mr = new CMeanRule();
	SGMatrix<float64_t> ensemble_matrix(num_vectors, num_classifiers);
	SGVector<float64_t> expected(num_vectors);

	generate_random_ensemble_matrix(ensemble_matrix);

	/* calculate expected values */
	for(index_t i = 0; i < ensemble_matrix.num_rows; i++)
	{
		SGVector<float64_t> rv = ensemble_matrix.get_row_vector(i);
		expected[i] = SGVector<float64_t>::sum(rv, ensemble_matrix.num_cols);
	}

	scale<linalg::Backend::NATIVE>(expected, 1/(float64_t)num_classifiers);

	SGVector<float64_t> combined = mr->combine(ensemble_matrix);

	EXPECT_EQ(num_vectors, combined.vlen);

	for (index_t i = 0; i < combined.vlen; ++i)
		EXPECT_DOUBLE_EQ(expected[i], combined[i]);

	SG_UNREF(mr);
}

TEST(MeanRule, combine_vector)
{
	int32_t vector_size = 20;
	CMeanRule* mr = new CMeanRule();
	SGVector<float64_t> test_labels(vector_size);
	test_labels.random(0.0, 50.0);

	float64_t expected = SGVector<float64_t>::sum(test_labels);
	expected /= (float64_t)vector_size;
	float64_t combined = mr->combine(test_labels);

	SG_UNREF(mr);

	EXPECT_DOUBLE_EQ(expected, combined);
}

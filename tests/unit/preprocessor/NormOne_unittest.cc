/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/NormOne.h>

using namespace shogun;

class NormOneTest : public ::testing::Test
{
public:
	NormOneTest()
	    : feats(nullptr),
	      transformer(std::make_shared<NormOne>())
	{
		matrix = SGMatrix<float64_t>(data, num_features, num_vectors, false);
		auto cloned_matrix = matrix.clone();
		feats = std::make_shared<DenseFeatures<float64_t>>(cloned_matrix);
	}

protected:
	float64_t data[6] = {1, 2, 3, 4, 5, 6};
	float64_t norm[2] = {std::sqrt(1 + 2 * 2 + 3 * 3),
	                     std::sqrt(4 * 4 + 5 * 5 + 6 * 6)};
	SGMatrix<float64_t> matrix;

	int32_t num_vectors = 2;
	int32_t num_features = 3;

	std::shared_ptr<DenseFeatures<float64_t>> feats;
	std::shared_ptr<NormOne> transformer;
};

TEST_F(NormOneTest, transform)
{
	transformer->fit(feats);
	feats =
	    transformer->transform(feats)->as<DenseFeatures<float64_t>>();

	ASSERT_EQ(feats->get_num_vectors(), num_vectors);

	for (auto i : range(num_vectors))
	{
		SGVector<float64_t> v = feats->get_feature_vector(i);
		ASSERT_EQ(v.vlen, num_features);
		for (auto j : range(v.vlen))
		{
			EXPECT_DOUBLE_EQ(v[j], matrix(j, i) / norm[i]);
		}
	}
}

TEST_F(NormOneTest, apply_to_vector)
{
	transformer->fit(feats);
	feats->add_preprocessor(transformer);

	for (auto i : range(num_vectors))
	{
		SGVector<float64_t> result = feats->get_feature_vector(i);
		ASSERT_EQ(result.vlen, num_features);
		for (auto j : range(result.vlen))
		{
			EXPECT_DOUBLE_EQ(result[j], matrix(j, i) / norm[i]);
		}
	}
}

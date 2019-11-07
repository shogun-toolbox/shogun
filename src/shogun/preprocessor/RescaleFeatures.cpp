/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Bjoern Esser
 */

#include <algorithm>
#include <shogun/base/range.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/RescaleFeatures.h>

using namespace shogun;

RescaleFeatures::RescaleFeatures() : DensePreprocessor<float64_t>()
{
	register_parameters();
}

RescaleFeatures::~RescaleFeatures()
{
	cleanup();
}

void RescaleFeatures::fit(std::shared_ptr<Features> features)
{
	if (m_fitted)
		cleanup();

	auto simple_features = features->as<DenseFeatures<float64_t>>();
	int32_t num_examples = simple_features->get_num_vectors();
	int32_t num_features = simple_features->get_num_features();
	require(
	    num_examples > 1, "number of feature vectors should be at least 2!");

	io::info("Extracting min and range values for each feature");

	m_min = SGVector<float64_t>(num_features);
	m_range = SGVector<float64_t>(num_features);
	auto feature_matrix = simple_features->get_feature_matrix();
	for (index_t i = 0; i < num_features; i++)
	{
		SGVector<float64_t> vec = feature_matrix.get_row_vector(i);
		float64_t cur_min = vec[0];
		float64_t cur_max = vec[0];

		/* find the max and min values in one loop */
		for (index_t j = 1; j < vec.vlen; j++)
		{
			cur_min = std::min(vec[j], cur_min);
			cur_max = std::max(vec[j], cur_max);
		}

		/* only rescale if range > 0 */
		if ((cur_max - cur_min) > 0)
		{
			m_min[i] = cur_min;
			m_range[i] = 1.0 / (cur_max - cur_min);
		}
		else
		{
			m_min[i] = 0.0;
			m_range[i] = 1.0;
		}
	}

	m_fitted = true;
}

void RescaleFeatures::cleanup()
{
	m_fitted = false;
}

SGMatrix<float64_t>
RescaleFeatures::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	assert_fitted();

	ASSERT(matrix.num_rows == m_min.vlen);

	for (auto i : range(matrix.num_cols))
	{
		auto vec = matrix.get_column(i);
		linalg::add(vec, m_min, vec, 1.0, -1.0);
		linalg::element_prod(vec, m_range, vec);
	}

	return matrix;
}

SGVector<float64_t> RescaleFeatures::apply_to_feature_vector(SGVector<float64_t> vector)
{
	assert_fitted();
	ASSERT(m_min.vlen == vector.vlen);

	float64_t* ret = SG_MALLOC(float64_t, vector.vlen);
	SGVector<float64_t>::add(ret, 1.0, vector.vector, -1.0, m_min.vector, vector.vlen);
	for (index_t i = 0; i < vector.vlen; i++) {
		ret[i] *= m_range[i];
	}

	return SGVector<float64_t>(ret,vector.vlen);
}

void RescaleFeatures::register_parameters()
{
	SG_ADD(&m_min, "min", "minimum values of each feature");
	SG_ADD(&m_range, "range", "Reciprocal of the range of each feature");
}

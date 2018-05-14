/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Sanuj Sharma, Sergey Lisitsyn, 
 *          Viktor Gal
 */

#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/features/Features.h>

using namespace shogun;

CNormOne::CNormOne()
: CDensePreprocessor<float64_t>()
{
}

CNormOne::~CNormOne()
{
}

/// clean up allocated memory
void CNormOne::cleanup()
{
}

/// initialize preprocessor from file
bool CNormOne::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CNormOne::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
SGMatrix<float64_t> CNormOne::apply_to_feature_matrix(CFeatures* features)
{
	auto feature_matrix =
	    features->as<CDenseFeatures<float64_t>>()->get_feature_matrix();

	for (int32_t i=0; i<feature_matrix.num_cols; i++)
	{
		SGVector<float64_t> vec(&(feature_matrix.matrix[i*feature_matrix.num_rows]), feature_matrix.num_rows, false);
		float64_t norm = std::sqrt(linalg::dot(vec, vec));
		SGVector<float64_t>::scale_vector(1.0/norm, vec, feature_matrix.num_rows);
	}
	return feature_matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CNormOne::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* normed_vec = SG_MALLOC(float64_t, vector.vlen);
	float64_t norm = std::sqrt(linalg::dot(vector, vector));

	for (int32_t i=0; i<vector.vlen; i++)
		normed_vec[i]=vector.vector[i]/norm;

	return SGVector<float64_t>(normed_vec,vector.vlen);
}

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Sanuj Sharma, Sergey Lisitsyn, 
 *          Viktor Gal
 */

#include <shogun/base/range.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/NormOne.h>

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

SGMatrix<float64_t> CNormOne::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	for (auto i : range(matrix.num_cols))
	{
		auto vec = matrix.get_column(i);
		auto norm = linalg::norm(vec);
		linalg::scale(vec, vec, 1.0 / norm);
	}
	return matrix;
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

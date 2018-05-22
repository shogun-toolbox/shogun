/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Evgeniy Andreev
 */

#include <shogun/base/range.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/LogPlusOne.h>

using namespace shogun;

CLogPlusOne::CLogPlusOne()
: CDensePreprocessor<float64_t>()
{
}


CLogPlusOne::~CLogPlusOne()
{
}

/// clean up allocated memory
void CLogPlusOne::cleanup()
{
}

/// initialize preprocessor from file
bool CLogPlusOne::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CLogPlusOne::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

SGMatrix<float64_t> CLogPlusOne::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	for (auto j : range(matrix.num_cols))
	{
		for (auto i : range(matrix.num_rows))
			matrix(i, j) = std::log(matrix(i, j) + 1.0);
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CLogPlusOne::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* log_vec = SG_MALLOC(float64_t, vector.vlen);

	for (int32_t i=0; i<vector.vlen; i++)
		log_vec[i] = std::log(vector.vector[i] + 1.0);

	return SGVector<float64_t>(log_vec,vector.vlen);
}

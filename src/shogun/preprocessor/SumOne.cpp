/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Sergey Lisitsyn
 */

#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/SumOne.h>

using namespace shogun;

SumOne::SumOne()
: DensePreprocessor<float64_t>()
{
}

SumOne::~SumOne()
{
}

/// clean up allocated memory
void SumOne::cleanup()
{
}

/// initialize preprocessor from file
bool SumOne::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool SumOne::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

SGMatrix<float64_t> SumOne::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	for (int32_t i = 0; i < matrix.num_cols; i++)
	{
		auto vec = matrix.get_column(i);
		auto sum = linalg::sum(vec);
		linalg::scale(vec, vec, 1.0 / sum);
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> SumOne::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* normed_vec = SG_MALLOC(float64_t, vector.vlen);
	float64_t sum = SGVector<float64_t>::sum(vector.vector, vector.vlen);

	for (int32_t i=0; i<vector.vlen; i++)
		normed_vec[i]=vector.vector[i]/sum;

	return SGVector<float64_t>(normed_vec,vector.vlen);
}

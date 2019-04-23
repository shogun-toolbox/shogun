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
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/NormOne.h>

using namespace shogun;

NormOne::NormOne()
: DensePreprocessor<float64_t>()
{
}

NormOne::~NormOne()
{
}

/// clean up allocated memory
void NormOne::cleanup()
{
}

/// initialize preprocessor from file
bool NormOne::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool NormOne::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

SGMatrix<float64_t> NormOne::apply_to_matrix(SGMatrix<float64_t> matrix)
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
SGVector<float64_t> NormOne::apply_to_feature_vector(SGVector<float64_t> vector)
{
	return linalg::scale(vector, 1.0 / linalg::norm(vector));
}

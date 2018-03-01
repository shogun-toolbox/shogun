/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Evgeniy Andreev
 */

#include <shogun/preprocessor/LogPlusOne.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CLogPlusOne::CLogPlusOne()
: CDensePreprocessor<float64_t>()
{
}


CLogPlusOne::~CLogPlusOne()
{
}

/// initialize preprocessor from features
bool CLogPlusOne::init(CFeatures* features)
{
	ASSERT(features->get_feature_class()==C_DENSE)
	ASSERT(features->get_feature_type()==F_DREAL)

	return true;
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

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
SGMatrix<float64_t> CLogPlusOne::apply_to_feature_matrix(CFeatures* features)
{
	SGMatrix<float64_t> feature_matrix =
			((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	for (int32_t i=0; i<feature_matrix.num_cols; i++)
	{
		for (int32_t j=0; j<feature_matrix.num_rows; j++)
			feature_matrix.matrix[i * feature_matrix.num_rows + j] = std::log(
			    feature_matrix.matrix[i * feature_matrix.num_rows + j] + 1.0);
	}
	return feature_matrix;
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

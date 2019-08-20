/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shell Hu, Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/structure/DirectorStructuredModel.h>

#ifdef USE_SWIG_DIRECTORS

using namespace shogun;

CDirectorStructuredModel::CDirectorStructuredModel() : CStructuredModel()
{
}

CDirectorStructuredModel::~CDirectorStructuredModel()
{
}

int32_t CDirectorStructuredModel::get_dim() const
{
	error("Please implemement get_dim() in your target language before use");
	return 0;
}

CResultSet* CDirectorStructuredModel::argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training)
{
	error("Please implemement argmax(w,feat_idx,lab_idx,training) in your target language before use");
	return NULL;
}

SGVector< float64_t > CDirectorStructuredModel::get_joint_feature_vector(
		int32_t feat_idx,
		CStructuredData* y)
{
	error("Please implemement get_joint_feature_vector(feat_idx,y) in your target language before use");
	return SGVector<float64_t>();
}

float64_t CDirectorStructuredModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	error("Please implemement delta_loss(y1,y2) in your target language before use");
	return 0.0;
}

bool CDirectorStructuredModel::check_training_setup() const
{
	error("Please implemement check_trainig_setup() in your target language before use");
	return false;
}

void CDirectorStructuredModel::init_primal_opt(
		float64_t regularization,
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > & lb,
		SGVector< float64_t > & ub,
		SGMatrix< float64_t > & C)
{
	error("Please implemement init_primal_opt(regularization,A,a,B,b,lb,ub,C) in your target language before use");
}

void CDirectorStructuredModel::init_training()
{
	error("Please implemement init_training() in your target language before use");
}

#endif /* USE_SWIG_DIRECTORS */

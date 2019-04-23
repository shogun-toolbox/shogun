/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shell Hu, Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/structure/DirectorStructuredModel.h>

#ifdef USE_SWIG_DIRECTORS

using namespace shogun;

DirectorStructuredModel::DirectorStructuredModel() : StructuredModel()
{
}

DirectorStructuredModel::~DirectorStructuredModel()
{
}

int32_t DirectorStructuredModel::get_dim() const
{
	SG_ERROR("Please implemement get_dim() in your target language before use\n")
	return 0;
}

std::shared_ptr<ResultSet> DirectorStructuredModel::argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training)
{
	SG_ERROR("Please implemement argmax(w,feat_idx,lab_idx,training) in your target language before use\n")
	return NULL;
}

SGVector< float64_t > DirectorStructuredModel::get_joint_feature_vector(
		int32_t feat_idx,
		std::shared_ptr<StructuredData> y)
{
	SG_ERROR("Please implemement get_joint_feature_vector(feat_idx,y) in your target language before use\n")
	return SGVector<float64_t>();
}

float64_t DirectorStructuredModel::delta_loss(std::shared_ptr<StructuredData> y1, std::shared_ptr<StructuredData> y2)
{
	SG_ERROR("Please implemement delta_loss(y1,y2) in your target language before use\n")
	return 0.0;
}

bool DirectorStructuredModel::check_training_setup() const
{
	SG_ERROR("Please implemement check_trainig_setup() in your target language before use\n")
	return false;
}

void DirectorStructuredModel::init_primal_opt(
		float64_t regularization,
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > & lb,
		SGVector< float64_t > & ub,
		SGMatrix< float64_t > & C)
{
	SG_ERROR("Please implemement init_primal_opt(regularization,A,a,B,b,lb,ub,C) in your target language before use\n")
}

void DirectorStructuredModel::init_training()
{
	SG_ERROR("Please implemement init_training() in your target language before use\n")
}

#endif /* USE_SWIG_DIRECTORS */

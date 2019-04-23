#include <shogun/latent/DirectorLatentModel.h>

#ifdef USE_SWIG_DIRECTORS

using namespace shogun;

DirectorLatentModel::DirectorLatentModel() : LatentModel()
{

}

DirectorLatentModel::~DirectorLatentModel()
{

}

int32_t DirectorLatentModel::get_dim() const
{
	SG_ERROR("Please implemement get_dim() in your target language before use\n")
	return 0;
}

std::shared_ptr<DotFeatures> DirectorLatentModel::get_psi_feature_vectors()
{
	SG_ERROR("Please implemement get_psi_feature_vectors() in your target language before use\n")
	return NULL;
}

std::shared_ptr<Data> DirectorLatentModel::infer_latent_variable(const SGVector<float64_t>& w, index_t idx)
{
	SG_ERROR("Please implemement infer_latent_variable(w, idx) in your target language before use\n")
	return NULL;
}

void DirectorLatentModel::argmax_h(const SGVector<float64_t>& w)
{
	SG_ERROR("Please implemement argmax_h(w) in your target language before use\n")
}

#endif /* USE_SWIG_DIRECTORS */

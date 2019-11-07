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
	error("Please implemement get_dim() in your target language before use");
	return 0;
}

std::shared_ptr<DotFeatures> DirectorLatentModel::get_psi_feature_vectors()
{
	error("Please implemement get_psi_feature_vectors() in your target language before use");
	return NULL;
}

std::shared_ptr<Data> DirectorLatentModel::infer_latent_variable(const SGVector<float64_t>& w, index_t idx)
{
	error("Please implemement infer_latent_variable(w, idx) in your target language before use");
	return NULL;
}

void DirectorLatentModel::argmax_h(const SGVector<float64_t>& w)
{
	error("Please implemement argmax_h(w) in your target language before use");
}

#endif /* USE_SWIG_DIRECTORS */

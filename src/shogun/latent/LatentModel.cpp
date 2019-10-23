/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Thoralf Klein, Evgeniy Andreev, Soeren Sonnenburg
 */

#include <shogun/latent/LatentModel.h>
#include <shogun/labels/BinaryLabels.h>

#include <utility>

using namespace shogun;

LatentModel::LatentModel()
	: m_features(NULL),
	m_labels(NULL),
	m_do_caching(false),
	m_cached_psi(NULL)
{
	register_parameters();
}

LatentModel::LatentModel(std::shared_ptr<LatentFeatures> feats, std::shared_ptr<LatentLabels> labels, bool do_caching)
	: m_features(std::move(feats)),
	m_labels(std::move(labels)),
	m_do_caching(do_caching),
	m_cached_psi(NULL)
{
	register_parameters();


}

LatentModel::~LatentModel()
{



}

int32_t LatentModel::get_num_vectors() const
{
	return m_features->get_num_vectors();
}

void LatentModel::set_labels(std::shared_ptr<LatentLabels> labs)
{


	m_labels = std::move(labs);
}

std::shared_ptr<LatentLabels> LatentModel::get_labels() const
{

	return m_labels;
}

void LatentModel::set_features(std::shared_ptr<LatentFeatures> feats)
{


	m_features = std::move(feats);
}

void LatentModel::argmax_h(const SGVector<float64_t>& w)
{
	int32_t num = get_num_vectors();
	auto y = binary_labels(m_labels->get_labels());
	ASSERT(num > 0)
	ASSERT(num == m_labels->get_num_labels())

	// argmax_h only for positive examples
	for (int32_t i = 0; i < num; ++i)
	{
		if (y->get_label(i) == 1)
		{
			// infer h and set it for the argmax_h <w,psi(x,h)>
			auto latent_data = infer_latent_variable(w, i);
			m_labels->set_latent_label(i, latent_data);
		}
	}
}

void LatentModel::register_parameters()
{
	SG_ADD(&m_features, "features", "Latent features");
	SG_ADD(&m_labels, "labels", "Latent labels");
	SG_ADD(
	    &m_cached_psi, "cached_psi", "Cached PSI features after argmax_h");
	SG_ADD(
	    &m_do_caching, "do_caching",
	    "Indicate whether or not do PSI vector caching after argmax_h");
}


std::shared_ptr<LatentFeatures> LatentModel::get_features() const
{

	return m_features;
}

void LatentModel::cache_psi_features()
{
	if (m_do_caching)
	{
		if (m_cached_psi)

		m_cached_psi = this->get_psi_feature_vectors();

	}
}

std::shared_ptr<DotFeatures> LatentModel::get_cached_psi_features() const
{
	if (m_do_caching)
	{

		return m_cached_psi;
	}
	return NULL;
}

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/latent/LatentSVM.h>
#include <typeinfo>
#include <utility>

#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/latent/LatentModel.h>

using namespace shogun;

LatentSVM::LatentSVM()
	: LinearLatentMachine()
{
}

LatentSVM::LatentSVM(std::shared_ptr<LatentModel> model, float64_t C)
	: LinearLatentMachine(std::move(model), C)
{
}

LatentSVM::~LatentSVM()
{
}

std::shared_ptr<LatentLabels> LatentSVM::apply_latent()
{
	if (!m_model)
		error("LatentModel is not set!");

	if (m_model->get_num_vectors() < 1)
		return NULL;

	SGVector<float64_t> w = get_w();
	index_t num_examples = m_model->get_num_vectors();
	auto hs = std::make_shared<LatentLabels>(num_examples);
	auto ys = std::make_shared<BinaryLabels>(num_examples);
	hs->set_labels(ys);
	m_model->set_labels(hs);

	for (index_t i = 0; i < num_examples; ++i)
	{
		/* find h for the example */
		auto h = m_model->infer_latent_variable(w, i);
		hs->add_latent_label(h);
	}

	/* compute the y labels */
	auto x = m_model->get_psi_feature_vectors();
	x->dense_dot_range(ys->get_labels().vector, 0, num_examples, NULL, w.vector, w.vlen, 0.0);

	return hs;
}

float64_t LatentSVM::do_inner_loop(float64_t cooling_eps)
{
	auto ys = m_model->get_labels()->get_labels();
	auto feats = (m_model->get_caching() ?
			m_model->get_cached_psi_features() :
			m_model->get_psi_feature_vectors());
	SVMOcas svm(m_C, feats, ys);
	svm.set_epsilon(cooling_eps);
	svm.train();


	/* copy the resulting w */
	set_w(svm.get_w().clone());

	return svm.compute_primal_objective();
}

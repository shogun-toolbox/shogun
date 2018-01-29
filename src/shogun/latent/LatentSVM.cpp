/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/latent/LatentSVM.h>
#include <typeinfo>

#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/latent/LatentModel.h>

using namespace shogun;

CLatentSVM::CLatentSVM()
	: CLinearLatentMachine()
{
}

CLatentSVM::CLatentSVM(CLatentModel* model, float64_t C)
	: CLinearLatentMachine(model, C)
{
}

CLatentSVM::~CLatentSVM()
{
}

CLatentLabels* CLatentSVM::apply_latent()
{
	if (!m_model)
		SG_ERROR("LatentModel is not set!\n")

	if (m_model->get_num_vectors() < 1)
		return NULL;

	SGVector<float64_t> w = get_w();
	index_t num_examples = m_model->get_num_vectors();
	CLatentLabels* hs = new CLatentLabels(num_examples);
	CBinaryLabels* ys = new CBinaryLabels(num_examples);
	hs->set_labels(ys);
	m_model->set_labels(hs);

	for (index_t i = 0; i < num_examples; ++i)
	{
		/* find h for the example */
		CData* h = m_model->infer_latent_variable(w, i);
		hs->add_latent_label(h);
	}

	/* compute the y labels */
	CDotFeatures* x = m_model->get_psi_feature_vectors();
	x->dense_dot_range(ys->get_labels().vector, 0, num_examples, NULL, w.vector, w.vlen, 0.0);

	return hs;
}

float64_t CLatentSVM::do_inner_loop(float64_t cooling_eps)
{
	CLabels* ys = m_model->get_labels()->get_labels();
	CDotFeatures* feats = (m_model->get_caching() ?
			m_model->get_cached_psi_features() :
			m_model->get_psi_feature_vectors());
	CSVMOcas svm(m_C, feats, ys);
	svm.set_epsilon(cooling_eps);
	svm.train();
	SG_UNREF(ys);
	SG_UNREF(feats);

	/* copy the resulting w */
	set_w(svm.get_w().clone());

	return svm.compute_primal_objective();
}

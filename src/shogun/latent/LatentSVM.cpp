/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <typeinfo>

#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/latent/LatentSVM.h>

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

CLatentLabels* CLatentSVM::apply()
{
	if (!m_model)
		SG_ERROR("LatentModel is not set!\n");

	if (!features)
		return NULL;

	index_t num_examples = m_model->get_num_vectors();
	CLatentLabels* hs = new CLatentLabels(num_examples);
	CBinaryLabels* ys = new CBinaryLabels(num_examples);

	for (index_t i = 0; i < num_examples; ++i)
	{
		/* find h for the example */
		CData* h = m_model->infer_latent_variable(w, i);
		hs->set_latent_label(i, h);
		SGVector<float64_t> psi_feat = m_model->get_psi_feature_vector(i);

		/* calculate and set y for the example */
		float64_t y = w.dot(w.vector, psi_feat.vector, w.vlen);
		ys->set_label(i, y);
	}

	hs->set_labels(ys);

	return hs;
}

float64_t CLatentSVM::do_inner_loop(float64_t cooling_eps)
{
	CLabels* ys = m_model->get_labels()->get_labels();
	CSVMOcas svm(m_C, features, ys);
	svm.set_epsilon(cooling_eps);
	svm.train();
	SG_UNREF(ys);

	/* copy the resulting w */
	SGVector<float64_t> cur_w = svm.get_w();
	memcpy(w.vector, cur_w.vector, cur_w.vlen*sizeof(float64_t));

	return svm.compute_primal_objective();
}


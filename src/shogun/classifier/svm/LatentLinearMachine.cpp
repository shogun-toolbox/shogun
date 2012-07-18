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
#include <shogun/classifier/svm/LatentLinearMachine.h>
#include <shogun/features/LatentFeatures.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CLatentLinearMachine::CLatentLinearMachine()
	: CLinearMachine()
{
	init();
}

CLatentLinearMachine::CLatentLinearMachine(CLatentModel* model, float64_t C)
	: CLinearMachine()
{
	init();
	m_C1 = m_C2 = C;
	set_model(model);

	index_t feat_dim = m_model->get_dim();
	set_w(SGVector<float64_t> (feat_dim));

	/* create the temporal storage for PSI features */
	SGMatrix<float64_t> psi_m(feat_dim, m_model->get_num_vectors());
	((CDenseFeatures<float64_t>*)features)->set_feature_matrix(psi_m);
}

CLatentLinearMachine::~CLatentLinearMachine()
{
	SG_UNREF(m_model);
}

CLatentLabels* CLatentLinearMachine::apply()
{
	if (!features)
		return NULL;

	return NULL;
}

CLatentLabels* CLatentLinearMachine::apply(CFeatures* data)
{
	if (m_model == NULL)
		SG_ERROR("LatentModel is not set!\n");

	CLatentFeatures* lf = CLatentFeatures::obtain_from_generic(data);
	m_model->set_features(lf);
	index_t num_examples = m_model->get_num_vectors();
	CLatentLabels* labels = new CLatentLabels(num_examples);

	for (index_t i = 0; i < num_examples; ++i)
	{
		/* find h for the example */
		CLatentData* h = m_model->infer_latent_variable(w, i);
		labels->set_latent_label(i, h);
		SGVector<float64_t> psi_feat = m_model->get_psi_feature_vector(i);

		/* calculate and set y for the example */
		float64_t y = w.dot(w.vector, psi_feat.vector, w.vlen);
		labels->set_label(i, y);
	}

	return labels;
}

void CLatentLinearMachine::set_model(CLatentModel* latent_model)
{
	ASSERT(latent_model != NULL);
	SG_UNREF(m_model);
	SG_REF(latent_model);
	m_model = latent_model;
}

void CLatentLinearMachine::cache_psi_vectors()
{
	ASSERT(features != NULL);
	index_t num_vectors = features->get_num_vectors();
	for (index_t i = 0; i < num_vectors; ++i)
	{
		SGVector<float64_t> psi_feat =
			dynamic_cast<CDenseFeatures<float64_t>*>(features)->get_feature_vector(i);
		memcpy(psi_feat.vector, m_model->get_psi_feature_vector(i).vector, psi_feat.vlen*sizeof(float64_t));
	}
}

bool CLatentLinearMachine::train_machine(CFeatures* data)
{
	if (m_model == NULL)
		SG_ERROR("LatentModel is not set!\n");

	SG_DEBUG("Initialise PSI (x,h)\n");
	cache_psi_vectors();

	/*
	 * define variables for calculating the stopping
	 * criterion for the outer loop
	 */
	float64_t decrement = 0.0, primal_obj = 0.0, prev_po = 0.0;
	float64_t inner_eps = 0.5*m_C1*m_epsilon;
	bool stop = false;
	int32_t iter = 0;

	/* do CCCP */
	SG_DEBUG("Starting CCCP\n");
	while ((iter < 2)||(!stop&&(iter < m_max_iter)))
	{
		SG_DEBUG("iteration: %d\n", iter);
		/* do the SVM optimisation with fixed h* */
		SG_DEBUG("Do the inner loop of CCCP: optimize for w for fixed h*\n");

		/* TODO: change code that it can support structural SVM! */
		CLatentLabels* labels = m_model->get_labels();
		CSVMOcas svm(m_C1, features, labels);
		svm.set_epsilon(inner_eps);
		svm.train();
		SG_UNREF(labels);

		/* calculate the decrement */
		primal_obj = svm.compute_primal_objective();
		decrement = prev_po - primal_obj;
		prev_po = primal_obj;
		SG_DEBUG("decrement: %f\n", decrement);
		SG_DEBUG("primal objective: %f\n", primal_obj);

		/* check the stopping criterion */
		stop = (inner_eps < (0.5*m_C1*m_epsilon+1E-8)) && (decrement < m_C1*m_epsilon);

		inner_eps = -decrement*0.01;
		inner_eps = CMath::max(inner_eps, 0.5*m_C1*m_epsilon);
		SG_DEBUG("inner epsilon: %f\n", inner_eps);

		/* find argmaxH */
		SG_DEBUG("Find and set h_i = argmax_h (w, psi(x_i,h))\n");
		SGVector<float64_t> cur_w = svm.get_w();
		memcpy(w.vector, cur_w.vector, cur_w.vlen*sizeof(float64_t));
		m_model->argmax_h(w);

		SG_DEBUG("Recalculating PSI (x,h) with the new h variables\n");
		cache_psi_vectors();

		/* increment iteration counter */
		iter++;
	}

	return true;
}

void CLatentLinearMachine::init()
{
	m_C1 = m_C2 = 10.0;
	m_epsilon = 1E-3;
	m_max_iter = 400;
	features = new CDenseFeatures<float64_t> ();
	SG_REF(features);
	m_model = NULL;

	m_parameters->add(&m_C1, "C1",  "Cost constant 1.");
	m_parameters->add(&m_C2, "C2",  "Cost constant 2.");
	m_parameters->add(&m_epsilon, "epsilon", "Convergence precision.");
	m_parameters->add(&m_max_iter, "max_iter", "Maximum iterations.");
	m_parameters->add((CSGObject**)&m_model, "latent_model", "Latent Model.");
}


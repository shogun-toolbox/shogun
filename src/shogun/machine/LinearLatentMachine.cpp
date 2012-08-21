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

#include <shogun/machine/LinearLatentMachine.h>
#include <shogun/features/LatentFeatures.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CLinearLatentMachine::CLinearLatentMachine()
	: CLinearMachine()
{
	init();
}

CLinearLatentMachine::CLinearLatentMachine(CLatentModel* model, float64_t C)
	: CLinearMachine()
{
	init();
	m_C= C;
	set_model(model);

	index_t feat_dim = m_model->get_dim();
	set_w(SGVector<float64_t> (feat_dim));
}

CLinearLatentMachine::~CLinearLatentMachine()
{
	SG_UNREF(m_model);
}

CLatentLabels* CLinearLatentMachine::apply_latent(CFeatures* data)
{
	if (m_model == NULL)
		SG_ERROR("LatentModel is not set!\n");

	CLatentFeatures* lf = CLatentFeatures::obtain_from_generic(data);
	m_model->set_features(lf);

	return apply_latent();
}

void CLinearLatentMachine::set_model(CLatentModel* latent_model)
{
	ASSERT(latent_model != NULL);
	SG_UNREF(m_model);
	SG_REF(latent_model);
	m_model = latent_model;
}

void CLinearLatentMachine::cache_psi_vectors()
{
	set_features(m_model->get_psi_feature_vectors());
}

bool CLinearLatentMachine::train_machine(CFeatures* data)
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
	float64_t inner_eps = 0.5*m_C*m_epsilon;
	bool stop = false;
	int32_t iter = 0;

	/* do CCCP */
	SG_DEBUG("Starting CCCP\n");
	while ((iter < 2)||(!stop&&(iter < m_max_iter)))
	{
		SG_DEBUG("iteration: %d\n", iter);
		/* do the SVM optimisation with fixed h* */
		SG_DEBUG("Do the inner loop of CCCP: optimize for w for fixed h*\n");
		primal_obj = do_inner_loop(inner_eps);

		/* calculate the decrement */
		decrement = prev_po - primal_obj;
		prev_po = primal_obj;
		SG_DEBUG("decrement: %f\n", decrement);
		SG_DEBUG("primal objective: %f\n", primal_obj);

		/* check the stopping criterion */
		stop = (inner_eps < (0.5*m_C*m_epsilon+1E-8)) && (decrement < m_C*m_epsilon);

		inner_eps = -decrement*0.01;
		inner_eps = CMath::max(inner_eps, 0.5*m_C*m_epsilon);
		SG_DEBUG("inner epsilon: %f\n", inner_eps);

		/* find argmaxH */
		SG_DEBUG("Find and set h_i = argmax_h (w, psi(x_i,h))\n");
		m_model->argmax_h(w);

		SG_DEBUG("Recalculating PSI (x,h) with the new h variables\n");
		cache_psi_vectors();

		/* increment iteration counter */
		iter++;
	}

	return true;
}

void CLinearLatentMachine::init()
{
	m_C = 10.0;
	m_epsilon = 1E-3;
	m_max_iter = 400;
	features = new CDenseFeatures<float64_t> ();
	SG_REF(features);
	m_model = NULL;

	m_parameters->add(&m_C, "C",  "Cost constant.");
	m_parameters->add(&m_epsilon, "epsilon", "Convergence precision.");
	m_parameters->add(&m_max_iter, "max_iter", "Maximum iterations.");
	m_parameters->add((CSGObject**) &m_model, "latent_model", "Latent Model.");
}


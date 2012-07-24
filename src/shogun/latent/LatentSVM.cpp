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

float64_t CLatentSVM::do_inner_loop(float64_t cooling_eps)
{
	CLatentLabels* labels = m_model->get_labels();
	CSVMOcas svm(m_C, features, labels);
	svm.set_epsilon(cooling_eps);
	svm.train();
	SG_UNREF(labels);

	/* copy the resulting w */
	SGVector<float64_t> cur_w = svm.get_w();
	memcpy(w.vector, cur_w.vector, cur_w.vlen*sizeof(float64_t));

	return svm.compute_primal_objective();
}


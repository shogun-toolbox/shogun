/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <shogun/latent/LatentSOSVM.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>

using namespace shogun;

CLatentSOSVM::CLatentSOSVM()
	: CLinearLatentMachine()
{
	register_parameters();
	m_so_solver=NULL;
}

CLatentSOSVM::CLatentSOSVM(CLatentModel* model, CLinearStructuredOutputMachine* so_solver, float64_t C)
	: CLinearLatentMachine(model, C)
{
	register_parameters();
	set_so_solver(so_solver);
}

CLatentSOSVM::~CLatentSOSVM()
{
	SG_UNREF(m_so_solver);
}

CLatentLabels* CLatentSOSVM::apply_latent()
{
	return NULL;
}

void CLatentSOSVM::set_so_solver(CLinearStructuredOutputMachine* so)
{
	SG_REF(so);
	SG_UNREF(m_so_solver);
	m_so_solver = so;
}

float64_t CLatentSOSVM::do_inner_loop(float64_t cooling_eps)
{
	float64_t lambda = 1/m_C;
	CDualLibQPBMSOSVM* so = new CDualLibQPBMSOSVM();
	so->set_lambda(lambda);
	so->train();

	/* copy the resulting w */
	SGVector<float64_t> cur_w = so->get_w();
	memcpy(w.vector, cur_w.vector, cur_w.vlen*sizeof(float64_t));

	/* get the primal objective value */
	float64_t po = so->get_result().Fp;

	SG_UNREF(so);

	return po;
}

void CLatentSOSVM::register_parameters()
{
	m_parameters->add((CSGObject**)&m_so_solver, "so_solver", "Structured Output Solver.");
}


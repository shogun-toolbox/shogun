/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#include <shogun/structure/DualLibQPBMSOSVM2.h>
#include <shogun/structure/conbmrm.h>

using namespace shogun;

CDualLibQPBMSOSVM::CDualLibQPBMSOSVM()
:CLinearStructuredOutputMachine()
{
	init();
}

CDualLibQPBMSOSVM::CDualLibQPBMSOSVM(
		CStructuredModel*	model,
		CStructuredLabels*	labs,
		float64_t	_lambda,
		SGVector< float64_t >	W)
 : CLinearStructuredOutputMachine(model, labs)
{
	init();
	set_lambda(_lambda);

	// get dimension of w
	int32_t nDim=this->m_model->get_dim();

	// Check for initial solution
	if (W.vlen==0)
	{
		set_w(SGVector< float64_t >(nDim));
		get_w().zero();
	}
	else
	{
		ASSERT(W.size() == nDim);
		set_w(W);
	}
}

CDualLibQPBMSOSVM::~CDualLibQPBMSOSVM()
{
}

void CDualLibQPBMSOSVM::init()
{
	SG_ADD(&m_TolRel, "m_TolRel", "Relative tolerance", MS_AVAILABLE);
	SG_ADD(&m_TolAbs, "m_TolAbs", "Absolute tolerance", MS_AVAILABLE);
	SG_ADD(&m_BufSize, "m_BuffSize", "Size of CP Buffer", MS_AVAILABLE);
	SG_ADD(&m_lambda, "m_lambda", "Regularization constant lambda",
			MS_AVAILABLE);
	SG_ADD(&m_cleanICP, "m_cleanICP", "Inactive cutting plane removal flag",
			MS_AVAILABLE);
	SG_ADD(&m_cleanAfter,
			"m_cleanAfter",
			"Number of inactive iterations after which ICP will be removed",
			MS_AVAILABLE);
	SG_ADD(&m_K, "m_K", "Parameter K", MS_NOT_AVAILABLE);
	SG_ADD(&m_Tmax, "m_Tmax", "Parameter Tmax", MS_AVAILABLE);
	SG_ADD(&m_cp_models, "m_cp_models", "Number of cutting plane models",
			MS_AVAILABLE);

	set_TolRel(0.001);
	set_TolAbs(0.0);
	set_BufSize(1000);
	set_lambda(0.0);
	set_cleanICP(true);
	set_cleanAfter(10);
	set_K(0.4);
	set_Tmax(100);
	set_cp_models(1);
	set_solver(BMRM);
}

bool CDualLibQPBMSOSVM::train_machine(CFeatures* data)
{
	if (data)
		set_features(data);

	if (m_verbose)
	{
		if (m_helper != NULL)
			SG_UNREF(m_helper);

		m_helper = new CSOSVMHelper();
		SG_REF(m_helper);
	}

	// Initialize the model for training
	m_model->init_training();
	// call the solver
	switch(m_solver)
	{
		case BMRM: case PPBMRM: case P3BMRM:
			m_result=con_bmrm_solver(this, m_w.vector, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,
					, m_cp_models, m_verbose, m_solver);
		case NCBM:
			m_result=svm_ncbm_solver(this, m_w.vector, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, true /* convex */,
					true /* use line search*/, m_verbose);
			break;
		default:
			SG_ERROR("CDualLibQPBMSOSVM: m_solver=%d is not supported", m_solver);
	}

	if (m_result.exitflag>0)
		return true;
	else
		return false;
}

EMachineType CDualLibQPBMSOSVM::get_classifier_type()
{
	return CT_LIBQPSOSVM;
}


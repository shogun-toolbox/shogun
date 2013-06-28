/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/libbmrm.h>
#include <shogun/structure/libppbm.h>
#include <shogun/structure/libp3bm.h>
#include <shogun/structure/libncbm.h>

using namespace shogun;

CDualLibQPBMSOSVM::CDualLibQPBMSOSVM()
:CLinearStructuredOutputMachine()
{
	init();
}

CDualLibQPBMSOSVM::CDualLibQPBMSOSVM(
		CStructuredModel*   	model,
		CStructuredLabels*  	labs,
		float64_t           	_lambda,
		SGVector< float64_t >	W)
 : CLinearStructuredOutputMachine(model, labs)
{
	set_TolRel(0.001);
	set_TolAbs(0.0);
	set_BufSize(1000);
	set_lambda(_lambda);
	set_cleanICP(true);
	set_cleanAfter(10);
	set_K(0.4);
	set_Tmax(100);
	set_cp_models(1);
	set_verbose(false);
	set_solver(BMRM);

	// get dimension of w
	uint32_t nDim=this->m_model->get_dim();

	// Check for initial solution
	if (W.vlen==0)
	{
		m_w=SGVector< float64_t >(nDim);

		m_w.zero();
	}
	else
	{
		set_w(W);
	}

	init();
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
	SG_ADD(&m_verbose, "m_verbose", "Verbosity flag", MS_NOT_AVAILABLE);
}

bool CDualLibQPBMSOSVM::train_machine(CFeatures* data)
{
	if (data)
		set_features(data);

	// Initialize the model for training
	m_model->init_training();
	// call the solver
	switch(m_solver)
	{
		case BMRM:
			m_result=svm_bmrm_solver(this, m_w.vector, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,
					m_verbose);
			break;
		case PPBMRM:
			m_result=svm_ppbm_solver(this, m_w.vector, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,
					m_verbose);
			break;
		case P3BMRM:
			m_result=svm_p3bm_solver(this, m_w.vector, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,
					m_cp_models, m_verbose);
			break;
		case NCBM:
			m_result=svm_ncbm_solver(this, m_w.vector, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, true /* convex */,
					true /* use line search*/, m_verbose);
	}

	if (m_result.exitflag==1)
	{
		return true;
	}
	else
	{
		return false;
	}
}

EMachineType CDualLibQPBMSOSVM::get_classifier_type()
{
	return CT_LIBQPSOSVM;
}


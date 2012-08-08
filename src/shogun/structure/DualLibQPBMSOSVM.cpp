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
#include <shogun/structure/libppbm.h>
#include <shogun/structure/libp3bm.h>

using namespace shogun;

CDualLibQPBMSOSVM::CDualLibQPBMSOSVM()
:CLinearStructuredOutputMachine()
{
}

CDualLibQPBMSOSVM::CDualLibQPBMSOSVM(
		CStructuredModel*   model,
		CLossFunction*      loss,
		CStructuredLabels*  labs,
		CDotFeatures*       features,
		float64_t           lambda,
		CRiskFunction*      risk_function,
		CRiskData*          risk_data,
		float64_t*          W)
:CLinearStructuredOutputMachine(model, loss, labs, features)
{
	set_TolRel(0.001);
	set_TolAbs(0.0);
	set_BufSize(1000);
	set_lambda(lambda);
	set_cleanICP(true);
	set_cleanAfter(10);
	set_K(0.4);
	set_Tmax(100);
	set_nThreads(1);
	set_verbose(true);
	set_solver(BMRM);
	m_risk_function=risk_function;

	// risk data
	m_risk_data=risk_data;

	// get dimension of w
	uint32_t nDim=this->m_model->get_dim();

	if (W==0)
	{
		m_w=SGVector< float64_t >(nDim);
		m_w.zero();
	} else {
		m_w.clone_vector(W, (int32_t)nDim);
	}
}

CDualLibQPBMSOSVM::~CDualLibQPBMSOSVM()
{
	SG_UNREF(m_risk_data);
}

bool CDualLibQPBMSOSVM::train_machine(CFeatures* data)
{
	// call the solver
	switch(m_solver)
	{
		case BMRM:
			m_result=svm_bmrm_solver((void*)m_risk_data, m_w.vector, m_TolRel, m_TolAbs, m_lambda,
					m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,  m_verbose, m_risk_function);
			break;
		case PPBMRM:
			m_result=svm_ppbm_solver((void*)m_risk_data, m_w.vector, m_TolRel, m_TolAbs, m_lambda,
					m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,  m_verbose, m_risk_function);
			break;
		case P3BMRM:
			m_result=svm_p3bm_solver((void*)m_risk_data, m_w.vector, m_TolRel, m_TolAbs, m_lambda,
					m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax, m_nThreads, m_verbose, m_risk_function);
			break;
	}

	if (m_result.exitflag==1)
	{
		return true;
	} else {
		return false;
	}
}

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

using namespace shogun;

CDualLibQPBMSOSVM::CDualLibQPBMSOSVM()
:CLinearStructuredOutputMachine()
{
}

CDualLibQPBMSOSVM::CDualLibQPBMSOSVM(
		CStructuredModel* 	model,
		CLossFunction* 		loss,
		CStructuredLabels*	labs,
		CDotFeatures*		features,
		float64_t           lambda,
		CRiskFunction*      risk_function)
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
	m_risk_function=risk_function;
}

CDualLibQPBMSOSVM::~CDualLibQPBMSOSVM()
{
}

bool CDualLibQPBMSOSVM::train_machine(CFeatures* data)
{
	// get dimension of w
	uint32_t nDim=this->m_model->get_dim();

	// Initialize the weight vector
	m_w = SGVector< float64_t >(nDim);
	m_w.zero();

	bmrm_data_T bmrm_data;
	bmrm_data.X=this->m_features;
	bmrm_data.y=this->m_labels;
	bmrm_data.w_dim=nDim;

	// call the BMRM solver
	bmrm_return_value_T result = svm_bmrm_solver(&bmrm_data, m_w.vector, m_TolRel, m_TolAbs, m_lambda,
			m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,  m_risk_function);

	if (result.exitflag==1)
	{
		return true;
	} else {
		return false;
	}
}

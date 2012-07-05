/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/classifier/svm/PegasosSVM.h>
#include <shogun/optimization/pegasos/pegasos_optimize.h>

using namespace shogun;

CPegasosSVM::CPegasosSVM()
: CLinearMachine()
{
    init();
}

CPegasosSVM::CPegasosSVM(
	float64_t lambda, CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine()
{
	init();
	m_lambda = lambda;

	set_features(traindat);
	set_labels(trainlab);
}

void CPegasosSVM::init()
{
	m_lambda = 0.01;
	set_max_iterations();

	SG_ADD(&m_lambda, "lambda", "lambda regularization constant", MS_AVAILABLE);
	SG_ADD(&m_max_iterations, "max_iterations", "max number of iterations",
			MS_NOT_AVAILABLE);
}

CPegasosSVM::~CPegasosSVM()
{
}

bool CPegasosSVM::train_machine(CFeatures* data)
{
	if (data)
		set_features((CDotFeatures*)data);

	double obj_value = 0.0;
	double norm_value = 0.0;
	double loss_value = 0.0;

	int eta_rule_type = 0;
	int eta_constant = 0;
	int projection_rule = 0;
	double projection_constant = 0;

	w = CPegasos::Learn(features, ((CBinaryLabels*)m_labels)->get_labels(), features->get_dim_feature_space(), m_lambda,
	                    m_max_iterations, 1, 100, obj_value, norm_value, loss_value, 
	                    eta_rule_type, eta_constant, projection_rule, projection_constant);
	set_bias(0.0);

	return true;
}

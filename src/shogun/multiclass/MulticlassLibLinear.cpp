/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/classifier/svm/SVM_linear.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CMulticlassLibLinear::CMulticlassLibLinear() :
	CLinearMulticlassMachine()
{
	init_defaults();
}

CMulticlassLibLinear::CMulticlassLibLinear(float64_t C, CDotFeatures* features, CLabels* labs) :
	CLinearMulticlassMachine(ONE_VS_REST_STRATEGY,features,NULL,labs)
{
	init_defaults();
	set_C(C);
}

void CMulticlassLibLinear::init_defaults()
{
	set_C(1.0);
	set_epsilon(1e-2);
	set_max_iter(10000);
	set_use_bias(false);
	set_save_train_state(false);
	m_train_state = NULL;
}

void CMulticlassLibLinear::register_parameters()
{
	m_parameters->add(&m_C, "m_C", "regularization constant");
	m_parameters->add(&m_epsilon, "m_epsilon", "tolerance epsilon");
	m_parameters->add(&m_max_iter, "m_max_iter", "max number of iterations");
	m_parameters->add(&m_use_bias, "m_use_bias", "indicates whether bias should be used");
	m_parameters->add(&m_save_train_state, "m_save_train_state", "indicates whether bias should be used");
}

CMulticlassLibLinear::~CMulticlassLibLinear()
{
	reset_train_state();
}

bool CMulticlassLibLinear::train_machine(CFeatures* data)
{
	if (data)
		set_features((CDotFeatures*)data);

	int32_t num_vectors = m_features->get_num_vectors();
	int32_t num_classes = m_labels->get_num_classes();
	int32_t bias_n = m_use_bias ? 1 : 0;

	problem mc_problem;
	mc_problem.l = num_vectors;
	mc_problem.n = m_features->get_dim_feature_space() + bias_n;
	mc_problem.y = SG_MALLOC(int32_t, mc_problem.l);
	for (int32_t i=0; i<num_vectors; i++)
		mc_problem.y[i] = m_labels->get_int_label(i);

	mc_problem.x = m_features;
	mc_problem.use_bias = m_use_bias;

	if (!m_train_state)
		m_train_state = new mcsvm_state();

	float64_t* C = SG_MALLOC(float64_t, num_vectors);
	for (int32_t i=0; i<num_vectors; i++)
		C[i] = m_C;

	Solver_MCSVM_CS solver(&mc_problem,num_classes,C,m_epsilon,
	                       m_max_iter,m_max_train_time,m_train_state);
	solver.solve();

	clear_machines();
	m_machines = SGVector<CMachine*>(num_classes);
	for (int32_t i=0; i<num_classes; i++)
	{
		CLinearMachine* machine = new CLinearMachine();
		float64_t* cw = SG_MALLOC(float64_t, mc_problem.n);

		for (int32_t j=0; j<mc_problem.n-bias_n; j++)
			cw[j] = m_train_state->w[j*num_classes+i];

		machine->set_w(SGVector<float64_t>(cw,mc_problem.n-bias_n));

		if (m_use_bias)
			machine->set_bias(m_train_state->w[(mc_problem.n-bias_n)*num_classes+i]);

		m_machines[i] = machine;
	}

	if (!m_save_train_state)
		reset_train_state();

	SG_FREE(C);
	SG_FREE(mc_problem.y);

	return true;
}
#endif /* HAVE_LAPACK */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <lib/config.h>
#ifdef HAVE_LAPACK
#include <transfer/domain_adaptation/DomainAdaptationMulticlassLibLinear.h>
#include <labels/MulticlassLabels.h>

using namespace shogun;

CDomainAdaptationMulticlassLibLinear::CDomainAdaptationMulticlassLibLinear() :
	CMulticlassLibLinear()
{
	init_defaults();
}

CDomainAdaptationMulticlassLibLinear::CDomainAdaptationMulticlassLibLinear(
		float64_t target_C, CDotFeatures* target_features, CLabels* target_labels,
		CLinearMulticlassMachine* source_machine) :
	CMulticlassLibLinear(target_C,target_features,target_labels)
{
	init_defaults();

	set_source_machine(source_machine);
}

void CDomainAdaptationMulticlassLibLinear::init_defaults()
{
	m_train_factor = 1.0;
	m_source_bias = 0.5;
	m_source_machine = NULL;

	register_parameters();
}

float64_t CDomainAdaptationMulticlassLibLinear::get_source_bias() const
{
	return m_source_bias;
}

void CDomainAdaptationMulticlassLibLinear::set_source_bias(float64_t source_bias)
{
	m_source_bias = source_bias;
}

float64_t CDomainAdaptationMulticlassLibLinear::get_train_factor() const
{
	return m_train_factor;
}

void CDomainAdaptationMulticlassLibLinear::set_train_factor(float64_t train_factor)
{
	m_train_factor = train_factor;
}

CLinearMulticlassMachine* CDomainAdaptationMulticlassLibLinear::get_source_machine() const
{
	SG_REF(m_source_machine);
	return m_source_machine;
}

void CDomainAdaptationMulticlassLibLinear::set_source_machine(
		CLinearMulticlassMachine* source_machine)
{
	SG_REF(source_machine);
	SG_UNREF(m_source_machine);
	m_source_machine = source_machine;
}

void CDomainAdaptationMulticlassLibLinear::register_parameters()
{
	SG_ADD((CSGObject**)&m_source_machine, "source_machine", "source domain machine",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_train_factor, "train_factor", "factor of target domain regularization",
			MS_AVAILABLE);
	SG_ADD(&m_source_bias, "source_bias", "bias to source domain",
			MS_AVAILABLE);
}

CDomainAdaptationMulticlassLibLinear::~CDomainAdaptationMulticlassLibLinear()
{
}

SGMatrix<float64_t> CDomainAdaptationMulticlassLibLinear::obtain_regularizer_matrix() const
{
	ASSERT(get_use_bias()==false)
	int32_t n_classes = ((CMulticlassLabels*)m_source_machine->get_labels())->get_num_classes();
	int32_t n_features = ((CDotFeatures*)m_source_machine->get_features())->get_dim_feature_space();
	SGMatrix<float64_t> w0(n_classes,n_features);

	for (int32_t i=0; i<n_classes; i++)
	{
		SGVector<float64_t> w = ((CLinearMachine*)m_source_machine->get_machine(i))->get_w();
		for (int32_t j=0; j<n_features; j++)
			w0(j,i) = m_train_factor*w[j];
	}

	return w0;
}

CBinaryLabels* CDomainAdaptationMulticlassLibLinear::get_submachine_outputs(int32_t i)
{
	CBinaryLabels* target_outputs = CMulticlassMachine::get_submachine_outputs(i);
	CBinaryLabels* source_outputs = m_source_machine->get_submachine_outputs(i);
	int32_t n_target_outputs = target_outputs->get_num_labels();
	ASSERT(n_target_outputs==source_outputs->get_num_labels())
	SGVector<float64_t> result(n_target_outputs);
	for (int32_t j=0; j<result.vlen; j++)
		result[j] = (1-m_source_bias)*target_outputs->get_value(j) + m_source_bias*source_outputs->get_value(j);

	SG_UNREF(target_outputs);
	SG_UNREF(source_outputs);

	return new CBinaryLabels(result);
}
#endif /* HAVE_LAPACK */

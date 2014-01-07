/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2011 Christian Widmer
 * Copyright (C) 2007-2011 Max-Planck-Society
 */

#include <lib/config.h>

#ifdef USE_SVMLIGHT

#include <transfer/domain_adaptation/DomainAdaptationSVM.h>
#include <io/SGIO.h>
#include <labels/Labels.h>
#include <labels/BinaryLabels.h>
#include <labels/RegressionLabels.h>
#include <iostream>
#include <vector>

using namespace shogun;

CDomainAdaptationSVM::CDomainAdaptationSVM() : CSVMLight()
{
	init();
}

CDomainAdaptationSVM::CDomainAdaptationSVM(float64_t C, CKernel* k, CLabels* lab, CSVM* pre_svm, float64_t B_param) : CSVMLight(C, k, lab)
{
	init();
	init(pre_svm, B_param);
}

CDomainAdaptationSVM::~CDomainAdaptationSVM()
{
	SG_UNREF(presvm);
	SG_DEBUG("deleting DomainAdaptationSVM\n")
}


void CDomainAdaptationSVM::init(CSVM* pre_svm, float64_t B_param)
{
	REQUIRE(pre_svm != NULL, "Pre SVM should not be null");
	// increase reference counts
	SG_REF(pre_svm);

	this->presvm=pre_svm;
	this->B=B_param;
	this->train_factor=1.0;

	// set bias of parent svm to zero
	this->presvm->set_bias(0.0);

	// invoke sanity check
	is_presvm_sane();
}

bool CDomainAdaptationSVM::is_presvm_sane()
{
	if (!presvm) {
		SG_ERROR("presvm is null")
	}

	if (presvm->get_num_support_vectors() == 0) {
		SG_ERROR("presvm has no support vectors, please train first")
	}

	if (presvm->get_bias() != 0) {
		SG_ERROR("presvm bias not set to zero")
	}

	if (presvm->get_kernel()->get_kernel_type() != this->get_kernel()->get_kernel_type()) {
		SG_ERROR("kernel types do not agree")
	}

	if (presvm->get_kernel()->get_feature_type() != this->get_kernel()->get_feature_type()) {
		SG_ERROR("feature types do not agree")
	}

	return true;
}


bool CDomainAdaptationSVM::train_machine(CFeatures* data)
{

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")
		kernel->init(data, data);
	}

	if (m_labels->get_label_type() != LT_BINARY)
		SG_ERROR("DomainAdaptationSVM requires binary labels\n")

	int32_t num_training_points = get_labels()->get_num_labels();
	CBinaryLabels* labels = (CBinaryLabels*) get_labels();

	float64_t* lin_term = SG_MALLOC(float64_t, num_training_points);

	// grab current training features
	CFeatures* train_data = get_kernel()->get_lhs();

	// bias of parent SVM was set to zero in constructor, already contains B
	CBinaryLabels* parent_svm_out = presvm->apply_binary(train_data);

	// pre-compute linear term
	for (int32_t i=0; i<num_training_points; i++)
	{
		lin_term[i] = train_factor * B * labels->get_label(i) * parent_svm_out->get_label(i) - 1.0;
	}

	//set linear term for QP
	this->set_linear_term(SGVector<float64_t>(lin_term, num_training_points));

	//train SVM
	bool success = CSVMLight::train_machine();
	SG_UNREF(labels);

	ASSERT(presvm)

	return success;

}


CSVM* CDomainAdaptationSVM::get_presvm()
{
	SG_REF(presvm);
	return presvm;
}


float64_t CDomainAdaptationSVM::get_B()
{
	return B;
}


float64_t CDomainAdaptationSVM::get_train_factor()
{
	return train_factor;
}


void CDomainAdaptationSVM::set_train_factor(float64_t factor)
{
	train_factor = factor;
}


CBinaryLabels* CDomainAdaptationSVM::apply_binary(CFeatures* data)
{
	ASSERT(data)
	ASSERT(presvm->get_bias()==0.0)

	int32_t num_examples = data->get_num_vectors();

	CBinaryLabels* out_current = CSVMLight::apply_binary(data);

	// recursive call if used on DomainAdaptationSVM object
	CBinaryLabels* out_presvm = presvm->apply_binary(data);

	// combine outputs
	SGVector<float64_t> out_combined(num_examples);
	for (int32_t i=0; i<num_examples; i++)
	{
		out_combined[i] = out_current->get_value(i) + B*out_presvm->get_value(i);
	}
	SG_UNREF(out_current);
	SG_UNREF(out_presvm);

	return new CBinaryLabels(out_combined);

}

void CDomainAdaptationSVM::init()
{
	presvm = NULL;
	B = 0;
	train_factor = 1.0;

	SG_ADD((CSGObject**) &presvm, "presvm", "SVM to regularize against.",
			MS_NOT_AVAILABLE);
	SG_ADD(&B, "B", "regularization parameter B.", MS_AVAILABLE);
	SG_ADD(&train_factor, "train_factor",
			"flag to switch off regularization in training.", MS_AVAILABLE);
}

#endif //USE_SVMLIGHT

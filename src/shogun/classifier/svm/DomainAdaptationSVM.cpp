/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2011 Christian Widmer
 * Copyright (C) 2007-2011 Max-Planck-Society
 */

#include <shogun/lib/config.h>

#ifdef USE_SVMLIGHT

#include <shogun/classifier/svm/DomainAdaptationSVM.h>
#include <shogun/io/io.h>
#include <iostream>
#include <vector>

using namespace shogun;

CDomainAdaptationSVM::CDomainAdaptationSVM() : CSVMLight()
{
}

CDomainAdaptationSVM::CDomainAdaptationSVM(float64_t C, CKernel* k, CLabels* lab, CSVM* pre_svm, float64_t B_param) : CSVMLight(C, k, lab)
{
	init();
	init(pre_svm, B_param);
}

CDomainAdaptationSVM::~CDomainAdaptationSVM()
{
	SG_UNREF(presvm);
	SG_DEBUG("deleting DomainAdaptationSVM\n");
}


void CDomainAdaptationSVM::init(CSVM* pre_svm, float64_t B_param)
{
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
		SG_ERROR("presvm is null");
	}

	if (presvm->get_num_support_vectors() == 0) {
		SG_ERROR("presvm has no support vectors, please train first");
	}

	if (presvm->get_bias() != 0) {
		SG_ERROR("presvm bias not set to zero");
	}

	if (presvm->get_kernel()->get_kernel_type() != this->get_kernel()->get_kernel_type()) {
		SG_ERROR("kernel types do not agree");
	}

	if (presvm->get_kernel()->get_feature_type() != this->get_kernel()->get_feature_type()) {
		SG_ERROR("feature types do not agree");
	}

	return true;
}


bool CDomainAdaptationSVM::train(CFeatures* data)
{

	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		kernel->init(data, data);
	}

	int32_t num_training_points = get_labels()->get_num_labels();


	float64_t* lin_term = new float64_t[num_training_points];

	// grab current training features
	CFeatures* train_data = get_kernel()->get_lhs();

	// bias of parent SVM was set to zero in constructor, already contains B
	CLabels* parent_svm_out = presvm->apply(train_data);

	// pre-compute linear term
	for (int32_t i=0; i<num_training_points; i++)
	{
		lin_term[i] = train_factor * B * get_label(i) * parent_svm_out->get_label(i) - 1.0;
	}

	//set linear term for QP
	this->set_linear_term(SGVector<float64_t>(lin_term, num_training_points));

	delete[] lin_term;

	//train SVM
	bool success = CSVMLight::train();

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


CLabels* CDomainAdaptationSVM::apply(CFeatures* data)
{

    ASSERT(presvm->get_bias()==0.0);

    int32_t num_examples = data->get_num_vectors();

    CLabels* out_current = CSVMLight::apply(data);

    // recursive call if used on DomainAdaptationSVM object
    CLabels* out_presvm = presvm->apply(data);


    // combine outputs
    for (int32_t i=0; i!=num_examples; i++)
    {
        float64_t out_combined = out_current->get_label(i) + B*out_presvm->get_label(i);
        out_current->set_label(i, out_combined);
    }

    return out_current;

}

void CDomainAdaptationSVM::init()
{
	presvm = NULL;
	B = 0;
	train_factor = 1.0;

	m_parameters->add((CSGObject**) &presvm, "presvm",
					  "SVM to regularize against.");
	m_parameters->add(&B, "B", "regularization parameter B.");
	m_parameters->add(&train_factor,
			"train_factor", "flag to switch off regularization in training.");
}

#endif //USE_SVMLIGHT

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Widmer
 * Copyright (C) 2007-2009 Max-Planck-Society
 */

#include "classifier/svm/DA_SVM.h"
#include "lib/io.h"
#include <iostream>


#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT(CDA_SVM);
#endif //HAVE_BOOST_SERIALIZATION


CDA_SVM::CDA_SVM() : CSVMLight()
{
}

CDA_SVM::CDA_SVM(float64_t C, CKernel* k, CLabels* lab, CSVM* pre_svm, float64_t B_param) : CSVMLight(C, k, lab)
{

  init(pre_svm, B_param);

}

CDA_SVM::~CDA_SVM()
{
	SG_UNREF(presvm);
	SG_DEBUG("deleting DA_SVM\n");
}


void CDA_SVM::init(CSVM* pre_svm, float64_t B_param)
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

bool CDA_SVM::is_presvm_sane()
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


bool CDA_SVM::train()
{

  int32_t num_training_points = get_labels()->get_num_labels();


  float64_t* lin_term = new float64_t[num_training_points];


  // grab current training features
  CFeatures* train_data = get_kernel()->get_lhs();


  // bias of parent SVM was set to zero in constructor, already contains B
  CLabels* parent_svm_out = presvm->classify(train_data);


  // pre-compute linear term
  for (int32_t i=0; i!=num_training_points; i++)
  {

    lin_term[i] = - B*(get_label(i) * parent_svm_out->get_label(i)) - 1.0;

  }


  //set linear term for QP
  this->set_linear_term(lin_term, num_training_points);

  //train SVM
  bool success = CSVMLight::train();

  ASSERT(presvm)


  //clean up
  delete[] lin_term;

  return success;

}


CSVM* CDA_SVM::get_presvm()
{

  return presvm;

}


float64_t CDA_SVM::get_B()
{

  return B;

}



CLabels* CDA_SVM::classify(CFeatures* data)
{

    ASSERT(presvm->get_bias()==0.0);

    int32_t num_examples = data->get_num_vectors();

    CLabels* out_current = CSVMLight::classify(data);

    // recursive call if used on DA_SVM object
    CLabels* out_presvm = presvm->classify(data);


    // combine outputs
    for (int32_t i=0; i!=num_examples; i++)
    {

        float64_t out_combined = out_current->get_label(i) + B*out_presvm->get_label(i);
        out_current->set_label(i, out_combined);

    }

    return out_current;

}


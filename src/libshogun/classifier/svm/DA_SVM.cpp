/*
 * This program free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Widmer
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/DA_SVM.h"
#include "lib/io.h"
#include <iostream>

/*
#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT(CDA_SVM);
#endif //HAVE_BOOST_SERIALIZATION
*/

CDA_SVM::CDA_SVM() : CSVMLight()
{
}

/*
CDA_SVM::CDA_SVM(std::string presvm_fn, float64_t B) : CSVMLight()
{

  CSVMLight* presvm = new CSVMLight();
  presvm->fromFile(presvm_fn);
  init(presvm, B);

}
*/

CDA_SVM::CDA_SVM(float64_t C, CKernel* k, CLabels* lab, CSVM* presvm, float64_t B) : CSVMLight(C, k, lab)
{

  init(presvm, B);
}


void CDA_SVM::init(CSVM* presvm, float64_t B)
{

  //increase reference counts
  SG_REF(presvm);

  //TODO: do some sanity checking, here, check if presvm is trained and if features are of the same type
  this->presvm=presvm;

  //set bias of parent svm to zero
  this->B=B;
  this->trainFactor=1.0;

  this->presvm->set_bias(0.0);
}


CDA_SVM::~CDA_SVM()
{
	//SG_PRINT("deleting DA_SVM\n");
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

    lin_term[i] = - (get_label(i) * parent_svm_out->get_label(i)) - 1.0;

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

    int32_t num_examples = data->get_num_vectors();

    CLabels* out_current = CSVMLight::classify(data);
    CLabels* out_presvm = presvm->classify(data);

    // combine outputs
    for (int32_t i=0; i!=num_examples; i++)
    {

        float64_t out_combined = out_current->get_label(i) + B*out_presvm->get_label(i);
        out_current->set_label(i, out_combined);

    }

    return out_current;

}


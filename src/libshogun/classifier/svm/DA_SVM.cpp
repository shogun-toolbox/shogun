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
  this->B=B;
  this->trainFactor=1.0;

}

CDA_SVM::~CDA_SVM()
{
	//SG_PRINT("deleting DA_SVM\n");
}


bool CDA_SVM::train()
{

  int32_t num_training_points = get_labels()->get_num_labels();

  double* linear_term = new double[num_training_points];


  CKernel* tmp_kernel = presvm->get_kernel();


  tmp_kernel->init(tmp_kernel->get_lhs(), get_kernel()->get_lhs());

  for (int32_t i=0; i!=num_training_points; i++)
  {

    float64_t tmp=0;

    for (int32_t j=0; j!=presvm->get_num_support_vectors(); j++)
    {

      float64_t alpha = presvm->get_alpha(j);
      int32_t sv_idx = presvm->get_support_vector(j);
      tmp += alpha * tmp_kernel->kernel(sv_idx,i);

    }

    //convex combination
    //linear_term[i] =  -1.0;
    //p[idx] = p[idx] - weight*(tmp_lab[idx] * tmp)

    linear_term[i] = - B * (get_label(i) * tmp) - 1.0;

  }

  /*
  //output first ten elements
  std::cout << "pv[0:4]:";

  for (int32_t i=0; i!=4; i++) {

    std::cout << pv[i] << ", ";

  }

  std::cout << std::endl;
  */

  //output first ten elements
  std::cout << "alphas[0:4]:";

  for (int32_t i=0; i!=4; i++) {

    std::cout << presvm->get_alpha(i) << ", ";

  }

  //set linear term for QP
  this->set_linear_term(linear_term, num_training_points);

  //train SVM
  bool success = CSVMLight::train();

  ASSERT(presvm)


  //clean up
  delete[] linear_term;

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

    set_batch_computation_enabled(false);
    int32_t num_examples = data->get_num_vectors();

    CLabels* out_current = CSVMLight::classify(data);
    CLabels* out_presvm = presvm->classify(data);

    float64_t bias = presvm->get_bias();

    for (int32_t i=0; i!=num_examples; i++)
    {
        //combine outputs
        float64_t out_combined = out_current->get_label(i) + B*(out_presvm->get_label(i) - bias);

        out_current->set_label(i, out_combined);
    }

    delete[] out_presvm;

    return out_current;

}

/*
CLabels* CDA_SVM::classify(CLabels* result)
{
			SG_NOTIMPLEMENTED;
			return NULL;

  //for DA-SVM only works in non-batch mode
  set_batch_computation_enabled(false);

  //assumption is that kernel of postsvm has been initialized
  //with test features on RHS!
  CKernel* pre_kernel = presvm->get_kernel();

  std::cout << "initializing pre-kernel" << std::endl;

  ASSERT(pre_kernel);
  ASSERT(get_kernel());
  ASSERT(get_kernel()->get_rhs());
  ASSERT(pre_kernel->get_rhs());
  ASSERT(pre_kernel->get_lhs());

  pre_kernel->init(pre_kernel->get_lhs(), get_kernel()->get_rhs());

  return CSVM::classify(result);

}
*/
/*
float64_t CDA_SVM::classify_example(INT num)
{

  //std::cout << "DA_SVM::classify_example" << std::endl;

  ASSERT(CKernelMachine::get_kernel());

  //call classification method from superclass
  float64_t dist = CSVM::classify_example(num);

  //works recursively if several DA_SVMs are stacked into each other
  
  float64_t tmp_dist = presvm->classify_example(num) - presvm->get_bias();

  return dist + tmp_dist;
  

   
  for(INT i=0; i< presvm->get_num_support_vectors(); i++)
  {
	dist+= B*(presvm->get_kernel()->kernel(presvm->get_support_vector(i), num)* presvm->get_alpha(i));
  }
  
  //equivalent python code
  //testout=svm_regul.classify().get_labels()+B*(presvm.classify().get_labels()-presvm.get_bias())

  //std::cout << "i=" << num << " distance after using v: " << dist << std::endl;

  return dist-get_bias();
  
}
*/

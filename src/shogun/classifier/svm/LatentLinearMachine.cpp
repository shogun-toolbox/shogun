/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <typeinfo>

#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/classifier/svm/LatentLinearMachine.h>
#include <shogun/features/LatentFeatures.h>

using namespace shogun;

CLatentLinearMachine::CLatentLinearMachine () 
  : argMaxH (NULL)
{
  init ();
}

CLatentLinearMachine::CLatentLinearMachine (argMaxLatent usrArgMaxFunc)
  : argMaxH (NULL)
{
  setArgmax (usrArgMaxFunc);
  init ();
}

CLatentLinearMachine::~CLatentLinearMachine ()
{
}

CLatentLinearMachine::CLatentLinearMachine (float64_t C,
    CLatentFeatures* traindat,
    CLabels* trainlab)
  : argMaxH (NULL)
{
  ASSERT (traindat != NULL);
  ASSERT (trainlab != NULL);
  ASSERT (trainlab->is_two_class_labeling () == true);

  init ();
  m_C1 = m_C2 = C;
  set_features (traindat);
  set_labels (trainlab);
}

CLatentLabels* CLatentLinearMachine::apply ()
{

  return NULL;
}

CLatentLabels* CLatentLinearMachine::apply (CFeatures* data)
{
  try
  {
    CLatentFeatures* lf = dynamic_cast<CLatentFeatures*> (data);
  }
  catch (const std::bad_cast& e)
  {
    SG_ERROR ("This object is not a LatentFeatures object: %s\n", e.what ());
  }

  return NULL;
}

void CLatentLinearMachine::setArgmax (argMaxLatent usrArgMaxFunc)
{
  ASSERT (usrArgMaxFunc != NULL);
  argMaxH = usrArgMaxFunc;
}

CFeatures* CLatentLinearMachine::defaultArgMaxH (CLatentLinearMachine& llm,
    void* userData)
{
  SGVector<float64_t> w = llm.get_w ();
  CLatentFeatures* features =
    dynamic_cast<CLatentFeatures*> (llm.get_features ());
  CLabels* labels = llm.get_labels ();

  int32_t num = features->get_num_vectors();
  ASSERT (num > 0);
  ASSERT (w.vlen == features->get_dim_feature_space ());
  ASSERT (labels->is_two_class_labeling () == true);

  float64_t* out = SG_MALLOC (float64_t, num);
  features->dense_dot_range (out, 0, num, NULL, w.vector, w.vlen, 0.0);

  /* find the index of the feature that has the max value of the dot product */
  int32_t index = CMath::arg_max (out, 1, num);

  return NULL;
}


bool CLatentLinearMachine::train_machine (CFeatures* data)
{
  try
  {
    CLatentFeatures* lf = dynamic_cast<CLatentFeatures*> (data);

    /* do CCCP */
    SG_DEBUG ("Starting CCCP\n");
    float64_t decrement = 0.0, primal_obj = 0.0, prev_po = 0.0;
    float64_t inner_eps = 0.5*m_C1*m_epsilon;
    bool stop = false;
    int32_t iter = 0;
    while (!stop||(iter < m_max_iter)) 
    {
      SG_DEBUG ("iteration: %d\n", iter);
      /* do the SVM optimisation with fixed h* */
      SG_DEBUG ("Do the inner loop of CCCP: optimize for w for fixed h*");			
      /* TODO: change code that it can support structural SVM! */
      CSVMOcas svm (m_C1, lf, NULL);
      svm.set_epsilon (inner_eps);
      svm.train ();

      /* calculate the decrement */
      primal_obj = svm.compute_primal_objective ();
      decrement = prev_po - primal_obj;
      prev_po = primal_obj;
      SG_DEBUG ("decrement: %f\n", decrement);
      SG_DEBUG ("primal objective: %f\n", primal_obj);

      /* check the stopping criterion */
      stop = (inner_eps < 0.5*m_C1*m_epsilon+1E-8) && (decrement < m_C1*m_epsilon);

      inner_eps = -decrement*0.01;
      inner_eps = CMath::max (inner_eps, 0.5*m_C1*m_epsilon);
      SG_DEBUG ("inner epsilon: %f\n", inner_eps);

      /* find argmaxH */
      SG_DEBUG ("Find h* = argmax_h <w,psi(x,y,h)>\n");
      svm.get_w (w, w_dim);
      argMaxH (*this, NULL);

      /* fix the found h* as observed */
      SG_DEBUG ("Set the h* for all the examples and recalculate PSI(x,y,h)\n");
      for (int i = 0; i < lf->get_num_vectors () ; ++i)
      {

      }

      /* increment iteration counter */
      iter++;
    }
  }
  catch (const std::bad_cast& e)
  {
    SG_ERROR ("This object is not a LatentFeatures object: %s\n", e.what ());
  }

  return true;
}

EClassifierType CLatentLinearMachine::get_classifier_type ()
{
  return CT_LATENTSVM;
}

void CLatentLinearMachine::init ()
{
  m_C1 = m_C2 = 10.0;
  m_epsilon = 1E-3;
  m_max_iter = 400;

  if (argMaxH == NULL)
    setArgmax (defaultArgMaxH);

  m_parameters->add(&m_C1, "C1",  "Cost constant 1.");
  m_parameters->add(&m_C2, "C2",  "Cost constant 2.");
  m_parameters->add(&m_epsilon, "epsilon", "Convergence precision.");
  m_parameters->add(&m_max_iter, "max_iter", "Maximum iterations.");
}


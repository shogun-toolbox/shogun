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
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CLatentLinearMachine::CLatentLinearMachine ()
  : argmax_h (NULL),
    psi (NULL),
    infer (NULL)
{
  init ();
}

CLatentLinearMachine::~CLatentLinearMachine ()
{
  SG_UNREF (m_latent_feats);
}

CLatentLinearMachine::CLatentLinearMachine (float64_t C,
    CLatentFeatures* traindat,
    CLabels* trainlab,
    index_t psi_size)
  : argmax_h (NULL),
    psi (NULL),
    infer (NULL),
    m_latent_feats (traindat)
{
  ASSERT (traindat != NULL);
  ASSERT (trainlab != NULL);

  init ();
  m_C1 = m_C2 = C;
  m_psi_size = psi_size;

  set_labels (trainlab);
  set_w (SGVector<float64_t> (m_psi_size));

  /* create the temporal storage for PSI features */
  SGMatrix<float64_t> psi_m (m_psi_size, m_latent_feats->get_num_vectors ());
  ((CDenseFeatures<float64_t>*)features)->set_feature_matrix (psi_m);
}

CLatentLabels* CLatentLinearMachine::apply ()
{
  if (!features)
    return NULL;

  return NULL;
}

CLatentLabels* CLatentLinearMachine::apply (CFeatures* data)
{
  try
  {
    CLatentFeatures* lf = dynamic_cast<CLatentFeatures*> (data);
    int32_t num_examples = lf->get_num_vectors ();
    SGMatrix<float64_t> psi_matrix (m_psi_size, num_examples);
    CDenseFeatures<float64_t> psi_feats (psi_matrix);
    CLatentLabels* labels = new CLatentLabels (num_examples);

    for (int i = 0; i < num_examples; ++i)
    {
      CLatentData* x = lf->get_sample (i);
      CLatentData* h = infer (*this, x);
      labels->set_latent_label (i, h);
      SGVector<float64_t> psi_feat = psi_feats.get_feature_vector (i);

      psi (*this, x, h, psi_feat.vector);

      float64_t y = w.dot (w.vector, psi_feat.vector, w.vlen);
      labels->set_confidence (i, y);
    }

    return labels;
  }
  catch (const std::bad_cast& e)
  {
    SG_ERROR ("This object is not a LatentFeatures object: %s\n", e.what ());
  }

  return NULL;
}

void CLatentLinearMachine::set_argmax (argmax_func usr_argmax)
{
  ASSERT (usr_argmax != NULL);
  argmax_h = usr_argmax;
}

void CLatentLinearMachine::set_psi (psi_func usr_psi)
{
  ASSERT (usr_psi != NULL);
  psi = usr_psi;
}

void CLatentLinearMachine::default_argmax_h (CLatentLinearMachine& llm,
    void* userData)
{
  SGVector<float64_t> w = llm.get_w ();
  CLatentFeatures* features = llm.get_latent_features ();
  CLatentLabels* labels =
    dynamic_cast<CLatentLabels*> (llm.get_labels ());

  int32_t num = features->get_num_vectors ();
  ASSERT (num > 0);

  /* argmax_h only for positive examples */
  for (int i = 0; i < num; ++i)
  {
    if (labels->get_confidence (i) == 1)
    {
      /* infer h and set it for the argmax_h <w,psi(x,h)> */
      CLatentData* latent_data = llm.infer (llm, features->get_sample (i));
      labels->set_latent_label (i, latent_data);
    }
  }
  SG_UNREF (features);
}

void CLatentLinearMachine::set_infer (infer_func usr_infer)
{
  ASSERT (usr_infer != NULL);
  infer = usr_infer;
}

void CLatentLinearMachine::compute_psi ()
{
  ASSERT (features != NULL);
  int32_t num_vectors = features->get_num_vectors ();
  for (int i = 0; i < num_vectors; ++i)
  {
    SGVector<float64_t> psi_feat = dynamic_cast<CDenseFeatures<float64_t>*>(features)->get_feature_vector (i);
    CLatentData* label = (dynamic_cast<CLatentLabels*> (m_labels))->get_latent_label (i);
    CLatentData* x = m_latent_feats->get_sample (i);
    psi (*this, x, label, psi_feat.vector);
  }
}

bool CLatentLinearMachine::train_machine (CFeatures* data)
{
  if (psi == NULL)
    SG_ERROR ("The PSI function is not implemented!\n");

  if (infer == NULL)
    SG_ERROR ("The Infer function is not implemented!\n");

  try
  {
    SG_DEBUG ("Initialise PSI (x,h)\n");
    compute_psi ();

    /* do CCCP */
    SG_DEBUG ("Starting CCCP\n");
    float64_t decrement = 0.0, primal_obj = 0.0, prev_po = 0.0;
    float64_t inner_eps = 0.5*m_C1*m_epsilon;
    bool stop = false;
    int32_t iter = 0;
    while ((iter < 2)||(!stop&&(iter < m_max_iter)))
    {
      SG_DEBUG ("iteration: %d\n", iter);
      /* do the SVM optimisation with fixed h* */
      SG_DEBUG ("Do the inner loop of CCCP: optimize for w for fixed h*\n");
      /* TODO: change code that it can support structural SVM! */
      CSVMOcas svm (m_C1, features, m_labels);
      svm.set_epsilon (inner_eps);
      svm.train ();

      /* calculate the decrement */
      primal_obj = svm.compute_primal_objective ();
      decrement = prev_po - primal_obj;
      prev_po = primal_obj;
      SG_DEBUG ("decrement: %f\n", decrement);
      SG_DEBUG ("primal objective: %f\n", primal_obj);

      /* check the stopping criterion */
      stop = (inner_eps < (0.5*m_C1*m_epsilon+1E-8)) && (decrement < m_C1*m_epsilon);

      inner_eps = -decrement*0.01;
      inner_eps = CMath::max (inner_eps, 0.5*m_C1*m_epsilon);
      SG_DEBUG ("inner epsilon: %f\n", inner_eps);

      /* find argmaxH */
      SG_DEBUG ("Find and set h_i = argmax_h (w, psi(x_i,h))\n");
      SGVector<float64_t> cur_w = svm.get_w ();
      memcpy (w.vector, cur_w.vector, cur_w.vlen*sizeof (float64_t));
      argmax_h (*this, NULL);

      SG_DEBUG ("Recalculating PSI (x,h) with the new h variables\n");
      compute_psi ();

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

void CLatentLinearMachine::init ()
{
  m_C1 = m_C2 = 10.0;
  m_epsilon = 1E-3;
  m_max_iter = 400;
  features = new CDenseFeatures<float64_t> ();
  SG_REF (features);

  if (argmax_h == NULL)
    set_argmax (default_argmax_h);

  m_parameters->add(&m_C1, "C1",  "Cost constant 1.");
  m_parameters->add(&m_C2, "C2",  "Cost constant 2.");
  m_parameters->add(&m_epsilon, "epsilon", "Convergence precision.");
  m_parameters->add(&m_max_iter, "max_iter", "Maximum iterations.");
  m_parameters->add(&m_psi_size, "psi_size", "PSI feature vector dimension.");
  m_parameters->add((CSGObject**) &m_latent_feats, "latent_feats", "Latent features");
}


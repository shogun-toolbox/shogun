/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * LatentSVM.cpp
 * Written (W) 2012 Shogun Google Summer of Code Xiangyu
 * Mentor By Alexander and Sonney
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Time.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/Parallel.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Labels.h>

using namespace shogun;

CLatentSVM::CLatentSVM()
: CLinearMachine()
{
	init();
}

CLatentSVM::CLatentSVM(E_SVM_TYPE type)
: CLinearMachine()
{
	init();
	method=type;
}

CLatentSVM::CLatentSVM(
	float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine()
{
	init();
	C1=C;
	C2=C;

	set_features(traindat);
	set_labels(trainlab);
}


CLatentSVM::~CLatentSVM()
{
}

bool CLatentSVM::train_machine(CFeatures* data)
{
	SG_INFO("C=%f, epsilon=%f, bufsize=%d\n", get_C1(), get_epsilon(), bufsize);
	SG_DEBUG("use_bias = %i\n", get_bias_enabled()) ;

	ASSERT(labels);
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	ASSERT(features);
	ASSERT(labels->is_two_class_labeling());

	lab=labels->get_labels();
	w_dim=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	if (num_vec!=lab.vlen || num_vec<=0)
		SG_ERROR("num_vec=%d num_train_labels=%d\n", num_vec, lab.vlen);

	SG_FREE(w);
	w=SG_MALLOC(float64_t, w_dim);
	memset(w, 0, w_dim*sizeof(float64_t));

	/*to be continued
	 *
	 */

	return true;
}


static float64_t CLatentSVM::sprod_nn(float64_t *a, float64_t *b, uint32_t n) {
  float64_t ans=0.0;
  uint32_t i;
  for (i=1;i<n+1;i++) {
    ans+=a[i]*b[i];
  }
  return(ans);
}

void CLatentSVM::add_vector_nn(float64_t *w, float64_t *dense_x, uint32_t n, float64_t factor) {
  uint32_t i;
  for (i=1;i<n+1;i++) {
    w[i]+=factor*dense_x[i];
  }
}

/* find_cutting_plane
 * The function is based on the equation
 * g(wt) - g(w) <= (w-wt)*vt for all w
 */
float64_t CLatentSVM::find_cutting_plane(void* ptr, float64_t **fycache)
{
  CLatentSVM *lsvm = (CLatentSVM) ptr;
  int32_t nData=lsvm->get_num_vectors();
  uint32_t nDim = (uint32_t) o->w_dim;
  float64_t* W=lsvm->w;
  float64_t* oldW=lsvm->old_w;
  CDotFeatures* x = lsvm->features;
  uint32_t nDim=(uint32_t) lsvm->w_dim;
  float64_t* y = lsvm->lab.vector;

  long i;
  float64_t *f, *fy, *fybar, *lhs;
  int32_t ybar;
  float64_t hbar;
  float64_t lossval;
  SGVector *new_constraint;
  float64_t *fvec;

  f=SG_MALLOC(float64_t, nDim);
  fy=SG_MALLOC(float64_t, nDim);
  fybar=SG_MALLOC(float64_t, nDim);
  lhs=SG_MALLOC(float64_t, nDim);
  fvec=SG_MALLOC(float64_t, nDim);

  long l,k;

  WORD *words;

  /* find cutting plane */
  lhs = NULL;
  *margin = 0;
  for (i=0;i<nData;i++) {
    find_most_violated_constraint_marginrescaling(x[i],y[i], &ybar, &hbar,lsvm);
    /* get difference vector */
    fy = fycache[i];
    fybar = psi(x[i],ybar,hbar,lsvm);
    lossval = loss(y[i],ybar,hbar,lsvm);

    /* scale difference vector */
    for (f=fy;f;f=f->next) {
      f->factor*=1.0/nData;
    }
    for (f=fybar;f;f=f->next) {
      f->factor*=-1.0/nData;
    }
    /* add ybar to constraint */
    append_svector_list(fy,lhs);
    append_svector_list(fybar,fy);
    lhs = fybar;
    *margin+=lossval/nData;
  }

  add_list_nn(new_constraint,lhs);

  SGFREE(f);
  SGFREE(fy);
  SGFREE(fybar);
  SGFREE(lhs);
  SGFREE£¨fvec);

  return(new_constraint);
}



void CLatentSVM::init()
{
	use_bias=true;
	bufsize=3000;
	C1=1;
	C2=1;
	
	epsilon=1e-3;
	method=SVM_LATENT;
	w=NULL;
	old_w=NULL;
	tmp_a_buf=NULL;
	lab.destroy_vector();
	cp_value=NULL;
	cp_index=NULL;
	cp_nz_dims=NULL;
	cp_bias=NULL;
}

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WEIGHTEDCOMMWORDSTRINGKERNEL_H___
#define _WEIGHTEDCOMMWORDSTRINGKERNEL_H___

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "kernel/CommWordStringKernel.h"

class CCommWordStringKernel;

class CWeightedCommWordStringKernel: public CCommWordStringKernel
{
 public:
  CWeightedCommWordStringKernel(LONG size, bool use_sign, ENormalizationType normalization_=FULL_NORMALIZATION );
  CWeightedCommWordStringKernel(CStringFeatures<WORD>* l, CStringFeatures<WORD>* r, bool use_sign=false, ENormalizationType normalization_=FULL_NORMALIZATION, INT size=10);
  ~CWeightedCommWordStringKernel();

  virtual bool init(CFeatures* l, CFeatures* r);
  virtual void cleanup();

  virtual DREAL compute_optimized(INT idx);
  virtual void add_to_normal(INT idx, DREAL weight);
  
  // init WD weighting
  bool set_wd_weights();

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_WEIGHTEDCOMMWORDSTRING; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "WeightedCommWordString"; }
  inline virtual EFeatureType get_feature_type() { return F_WORD; }
  
 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  DREAL compute(INT idx_a, INT idx_b);

  INT degree;

  /// weights for each of the subkernels of degree 1...d
  DREAL* weights;
};
#endif

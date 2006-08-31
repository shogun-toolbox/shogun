/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CONSTKERNEL_H___
#define _CONSTKERNEL_H___

#include "lib/Mathmatics.h"
#include "lib/common.h"
#include "kernel/Kernel.h"
#include "features/Features.h"

class CConstKernel: public CKernel
{
 public:
  CConstKernel(DREAL c);
  ~CConstKernel() ;

  inline virtual void cleanup() { }

  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Const,...
  inline virtual EKernelType get_kernel_type() { return K_CONST; }

  /** return feature type the kernel can deal with
  */
  inline virtual EFeatureType get_feature_type()
  {
	  return F_ANY;
  }

  /** return feature class the kernel can deal with
  */
  inline virtual EFeatureClass get_feature_class()
  {
	  return C_ANY;
  }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "Const"; }

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  inline virtual DREAL compute(INT row, INT col)
  {
	  return const_value;
  }

 protected:
  DREAL const_value;
};
#endif

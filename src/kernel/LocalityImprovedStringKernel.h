/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LOCALITYIMPROVEDSTRINGKERNEL_H___
#define _LOCALITYIMPROVEDSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

class CLocalityImprovedStringKernel: public CStringKernel<CHAR>
{
public:
	CLocalityImprovedStringKernel(INT size, INT length, INT inner_degree,
		INT outer_degree);
	~CLocalityImprovedStringKernel();

  virtual bool init(CFeatures* l, CFeatures* r);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_LOCALITYIMPROVED; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "LocalityImproved" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  DREAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

 protected:
  INT length;
  INT inner_degree;
  INT outer_degree;
  CHAR* match;
};

#endif


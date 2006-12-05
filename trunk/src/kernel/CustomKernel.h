/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CUSTOMKERNEL_H___
#define _CUSTOMKERNEL_H___

#include "lib/Mathematics.h"
#include "lib/common.h"
#include "kernel/Kernel.h"
#include "features/Features.h"

class CCustomKernel: public CKernel
{
 public:
  CCustomKernel();
  ~CCustomKernel() ;

  virtual SHORTREAL* get_kernel_matrix(int &m, int &n, SHORTREAL* target=NULL);
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Custom,...
  inline virtual EKernelType get_kernel_type() { return K_CUSTOM; }

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
  virtual const CHAR* get_name() { return "Custom"; }

  // set kernel matrix (only elements from main diagonal and above)
  // from elements of maindiagonal and above (concat'd), while m is 
  // already given as diagonal
  bool set_diag_kernel_matrix_from_diag(const DREAL* m, int cols);

  // set kernel matrix (only elements from main diagonal and above)
  // from squared matrix
  bool set_diag_kernel_matrix_from_full(const DREAL* m, int cols);

  // set full kernel matrix from full kernel matrix
  bool set_full_kernel_matrix_from_full(const DREAL* m, int rows, int cols);

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  inline virtual DREAL compute(INT row, INT col)
  {
	  ASSERT(row < num_rows);
	  ASSERT(col < num_cols);

	  if (upper_diagonal)
	  {
		  if (row <= col)
			  return kmatrix[row*num_cols - row*(row+1)/2 + col];
		  else
			  return kmatrix[col*num_cols - col*(col+1)/2 + row];
	  }
	  else
		  return kmatrix[row*num_cols+col];
  }

 protected:
  SHORTREAL* kmatrix;
  INT num_rows;
  INT num_cols;
  bool upper_diagonal;
};
#endif

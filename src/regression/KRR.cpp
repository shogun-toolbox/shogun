/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Mikio L. Braun
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "regression/KRR.h"
#include "lib/lapack.h"
#include "lib/Mathematics.h"

CKRR::CKRR() : CKernelMachine()
{
  alpha = NULL;
  tau = 1e-6;
}

CKRR::CKRR(DREAL t, CKernel* k, CLabels* lab)
{
  tau = t;
  set_labels(lab);
  set_kernel(k);
  alpha = NULL;
}


CKRR::~CKRR()
{
  delete[] alpha;
}

bool CKRR::train()
{
  delete[] alpha;
  
  ASSERT(labels);
  ASSERT(kernel && kernel->has_features());

  // Get kernel matrix
  INT m = 0;
  INT n = 0;
  DREAL *K = kernel->get_kernel_matrix_real(m, n, NULL);
  ASSERT(K && m > 0 && n > 0);
  
  for(int i = 0; i < n; i++)
	K[i + i*n] += tau;

  // Get labels
  INT numlabels = 0;
  alpha = get_labels()->get_labels(numlabels);
  ASSERT(alpha && numlabels == n);
  
  dposv('U', n, 1, K, n, alpha, n);
  
  delete[] K;

  return false;
}

bool CKRR::load(FILE* srcfile)
{
	return false;
}

bool CKRR::save(FILE* dstfile)
{
	return false;
}

CLabels* CKRR::classify(CLabels* output)
{
  if (labels)
	{
	  ASSERT(output == NULL);
	  ASSERT(kernel);

	  // Get kernel matrix
	  INT m = 0;
	  INT n = 0;
	  DREAL *K = kernel->get_kernel_matrix_real(m, n, NULL);
	  ASSERT(K && m > 0 && n > 0);
	  DREAL *Yh = new DREAL[n];

	  // predict
	  dgemv('T', m, n, 1.0, K, m, alpha, 1, 0.0, Yh, 1);
	  
	  delete[] K;

	  output=new CLabels(n);

	  output->set_labels(Yh, n);

	  delete[] Yh;
	  
	  return output;
	}
  
  return NULL;
}

DREAL CKRR::classify_example(INT num)
{
  ASSERT(kernel);

  // Get kernel matrix
  INT m = 0;
  INT n = 0;
  // TODO: use get_kernel_column instead of computing the whole matrix!
  DREAL *K = kernel->get_kernel_matrix_real(m, n, NULL);
  ASSERT(K && m > 0 && n > 0);
  DREAL Yh;
  
  // predict
  Yh = CMath::dot(K + m*num, alpha, m);
  
  delete[] K;

  return Yh;
}

#endif

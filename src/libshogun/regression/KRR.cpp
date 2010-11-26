/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Mikio L. Braun
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "regression/KRR.h"
#include "lib/lapack.h"
#include "lib/Mathematics.h"

using namespace shogun;

CKRR::CKRR()
: CKernelMachine()
{
	alpha=NULL;
	tau=1e-6;
}

CKRR::CKRR(float64_t t, CKernel* k, CLabels* lab)
: CKernelMachine()
{
	tau=t;
	set_labels(lab);
	set_kernel(k);
	alpha=NULL;
}


CKRR::~CKRR()
{
	delete[] alpha;
}

bool CKRR::train(CFeatures* data)
{
	delete[] alpha;

	ASSERT(labels);
	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		kernel->init(data, data);
	}
	ASSERT(kernel && kernel->has_features());

	// Get kernel matrix
	int32_t m=0;
	int32_t n=0;
	float64_t *K = kernel->get_kernel_matrix<float64_t>(m, n, NULL);
	ASSERT(K && m>0 && n>0);

	for(int32_t i=0; i < n; i++)
		K[i+i*n]+=tau;

	// Get labels
	int32_t numlabels=0;
	alpha=labels->get_labels(numlabels);
	if (!alpha)
		SG_ERROR("No labels set\n");

	if (numlabels!=n)
	{
		SG_ERROR("Number of labels does not match number of kernel"
				" columns (num_labels=%d cols=%d\n", numlabels, n);
	}

	clapack_dposv(CblasRowMajor,CblasUpper, n, 1, K, n, alpha, n);

	delete[] K;
	return true;
}

bool CKRR::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CKRR::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

CLabels* CKRR::classify()
{
	ASSERT(kernel);

	// Get kernel matrix
	int32_t m=0;
	int32_t n=0;
	float64_t* K=kernel->get_kernel_matrix<float64_t>(m, n, NULL);
	ASSERT(K && m>0 && n>0);
	float64_t* Yh=new float64_t[n];

	// predict
	// K is symmetric, CblasColMajor is same as CblasRowMajor 
	// and used that way in the origin call:
	// dgemv('T', m, n, 1.0, K, m, alpha, 1, 0.0, Yh, 1);
	int m_int = (int) m;
	int n_int = (int) n;
	cblas_dgemv(CblasColMajor, CblasTrans, m_int, n_int, 1.0, (double*) K,
		m_int, (double*) alpha, 1, 0.0, (double*) Yh, 1);

	delete[] K;

	CLabels* output=new CLabels(n);
	output->set_labels(Yh, n);

	delete[] Yh;

	return output;
}

float64_t CKRR::classify_example(int32_t num)
{
	ASSERT(kernel);

	// Get kernel matrix
	int32_t m=0;
	int32_t n=0;
	// TODO: use get_kernel_column instead of computing the whole matrix!
	float64_t* K=kernel->get_kernel_matrix<float64_t>(m, n, NULL);
	ASSERT(K && m>0 && n>0);
	float64_t Yh;

	// predict
	Yh = CMath::dot(K + m*num, alpha, m);

	delete[] K;
	return Yh;
}

#endif

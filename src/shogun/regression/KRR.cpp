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

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/regression/KRR.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>

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
	SG_FREE(alpha);
}

bool CKRR::train_machine(CFeatures* data)
{
	SG_FREE(alpha);

	ASSERT(labels);
	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		kernel->init(data, data);
	}
	ASSERT(kernel && kernel->has_features());

	// Get kernel matrix
	SGMatrix<float64_t> kernel_matrix=kernel->get_kernel_matrix<float64_t>();
	int32_t n = kernel_matrix.num_cols;
	int32_t m = kernel_matrix.num_rows;
	ASSERT(kernel_matrix.matrix && m>0 && n>0);

	for(int32_t i=0; i < n; i++)
		kernel_matrix.matrix[i+i*n]+=tau;

	// Get labels
	if (!labels)
		SG_ERROR("No labels set\n");

	SGVector<float64_t> alpha_orig=labels->get_labels();

	alpha=CMath::clone_vector(alpha_orig.vector, alpha_orig.vlen);

	if (alpha_orig.vlen!=n)
	{
		SG_ERROR("Number of labels does not match number of kernel"
				" columns (num_labels=%d cols=%d\n", alpha_orig.vlen, n);
	}

	clapack_dposv(CblasRowMajor,CblasUpper, n, 1, kernel_matrix.matrix, n, alpha, n);

	SG_FREE(kernel_matrix.matrix);
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

CLabels* CKRR::apply()
{
	ASSERT(kernel);

	// Get kernel matrix
	SGMatrix<float64_t> kernel_matrix=kernel->get_kernel_matrix<float64_t>();
	int32_t n = kernel_matrix.num_cols;
	int32_t m = kernel_matrix.num_rows;
	ASSERT(kernel_matrix.matrix && m>0 && n>0);

	SGVector<float64_t> Yh(n);

	// predict
	// K is symmetric, CblasColMajor is same as CblasRowMajor 
	// and used that way in the origin call:
	// dgemv('T', m, n, 1.0, K, m, alpha, 1, 0.0, Yh, 1);
	int m_int = (int) m;
	int n_int = (int) n;
	cblas_dgemv(CblasColMajor, CblasTrans, m_int, n_int, 1.0, (double*) kernel_matrix.matrix,
		m_int, (double*) alpha, 1, 0.0, (double*) Yh.vector, 1);

	SG_FREE(kernel_matrix.matrix);

	return new CLabels(Yh);
}

float64_t CKRR::apply(int32_t num)
{
	ASSERT(kernel);

	// Get kernel matrix
	// TODO: use get_kernel_column instead of computing the whole matrix!
	SGMatrix<float64_t> kernel_matrix=kernel->get_kernel_matrix<float64_t>();
	int32_t n = kernel_matrix.num_cols;
	int32_t m = kernel_matrix.num_rows;
	ASSERT(kernel_matrix.matrix && m>0 && n>0);

	float64_t Yh;

	// predict
	Yh = CMath::dot(kernel_matrix.matrix + m*num, alpha, m);

	SG_FREE(kernel_matrix.matrix);
	return Yh;
}

#endif

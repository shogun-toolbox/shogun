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
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>

#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;

CKernelRidgeRegression::CKernelRidgeRegression()
: CKernelMachine()
{
	init();
}

CKernelRidgeRegression::CKernelRidgeRegression(float64_t tau, CKernel* k, CLabels* lab, ETrainingType m)
: CKernelMachine()
{
	init();

	m_tau=tau;
	set_labels(lab);
	set_kernel(k);
	m_train_func=m;
}

void CKernelRidgeRegression::init()
{
	m_tau=1e-6;
	m_epsilon=0.0001;
	SG_ADD(&m_tau, "tau", "Regularization parameter", MS_AVAILABLE);
}

bool CKernelRidgeRegression::train_machine_pinv()
{
	// Get kernel matrix
	SGMatrix<float64_t> kernel_matrix=kernel->get_kernel_matrix<float64_t>();
	int32_t n = kernel_matrix.num_cols;
	int32_t m = kernel_matrix.num_rows;
	ASSERT(kernel_matrix.matrix && m>0 && n>0)

	for(int32_t i=0; i < n; i++)
		kernel_matrix.matrix[i+i*n]+=m_tau;

	/* re-set alphas of kernel machine */
	m_alpha=((CRegressionLabels*) m_labels)->get_labels_copy();

	/* tell kernel machine that all alphas are needed as'support vectors' */
	m_svs=SGVector<index_t>(m_alpha.vlen);
	linalg::range_fill<linalg::Backend::NATIVE>(m_svs,m_svs.vlen);

	if (get_alphas().vlen!=n)
	{
		SG_ERROR("Number of labels does not match number of kernel"
				" columns (num_labels=%d cols=%d\n", m_alpha.vlen, n);
	}

	clapack_dposv(CblasRowMajor,CblasUpper, n, 1, kernel_matrix.matrix, n,
			m_alpha.vector, n);

	return true;
}

bool CKernelRidgeRegression::train_machine_gs()
{
	int32_t n = kernel->get_num_vec_rhs();
	int32_t m = kernel->get_num_vec_lhs();
	ASSERT(m>0 && n>0)

	// re-set alphas of kernel machine
	SGVector<float64_t> b;
	float64_t alpha_old;

	b=((CRegressionLabels*) m_labels)->get_labels_copy();
	m_alpha=((CRegressionLabels*) m_labels)->get_labels_copy();
	m_alpha.zero();

	// tell kernel machine that all alphas are needed as 'support vectors'
	m_svs=SGVector<index_t>(m_alpha.vlen);
	linalg::range_fill<linalg::Backend::NATIVE>(m_svs,m_svs.vlen);

	if (get_alphas().vlen!=n)
	{
		SG_ERROR("Number of labels does not match number of kernel"
				" columns (num_labels=%d cols=%d\n", m_alpha.vlen, n);
	}

	// Gauss-Seidel iterative method
	float64_t sigma, err, d;
	bool flag=true;
	while(flag)
	{
		err=0.0;
		for(int32_t i=0; i<n; i++)
		{
			sigma=b[i];
			for(int32_t j=0; j<n; j++)
				if (i!=j)
					sigma-=kernel->kernel(j, i)*m_alpha[j];
			alpha_old=m_alpha[i];
			m_alpha[i]=sigma/(kernel->kernel(i, i)+m_tau);
			d=fabs(alpha_old-m_alpha[i]);
			if(d>err)
				err=d;
		}
		if (err<=m_epsilon)
			flag=false;
	}

	return true;
}

bool CKernelRidgeRegression::train_machine(CFeatures *data)
{
	if (!m_labels)
		SG_ERROR("No labels set\n")

	if (m_labels->get_label_type() != LT_REGRESSION)
		SG_ERROR("Real labels needed for kernel ridge regression.\n")

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")
		kernel->init(data, data);
	}
	ASSERT(kernel && kernel->has_features())

	switch (m_train_func)
	{
		case PINV:
			return train_machine_pinv();
			break;
		case GS:
			return train_machine_gs();
			break;
		default:
			return train_machine_pinv();
			break;
	}
}

bool CKernelRidgeRegression::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CKernelRidgeRegression::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}
#endif

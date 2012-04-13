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
    set_epsilon(0.0001);
    m_train_func=getTrainFunction(m);
}

void CKernelRidgeRegression::init()
{
	m_tau=1e-6;

	SG_ADD(&m_tau, "tau", "Regularization parameter", MS_AVAILABLE);
}

CKernelRidgeRegression::train_method CKernelRidgeRegression::getTrainFunction(ETrainingType id)
{
    switch (id)
    {
        case PINV:
            return &CKernelRidgeRegression::train_machine_pinv;
        break;
        case GS:
            return &CKernelRidgeRegression::train_machine_gs;
        break;
        default:
            return &CKernelRidgeRegression::train_machine_pinv;
        break;
    }
}

bool CKernelRidgeRegression::train_machine_pinv(CFeatures* data)
{
	if (!m_labels)
		SG_ERROR("No labels set\n");

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
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
		kernel_matrix.matrix[i+i*n]+=m_tau;

	/* re-set alphas of kernel machine */
	m_alpha.destroy_vector();
	m_alpha=m_labels->get_labels_copy();

	/* tell kernel machine that all alphas are needed as'support vectors' */
	m_svs.destroy_vector();
	m_svs=SGVector<index_t>(m_alpha.vlen);
	m_svs.range_fill();

	if (get_alphas().vlen!=n)
	{
		SG_ERROR("Number of labels does not match number of kernel"
				" columns (num_labels=%d cols=%d\n", m_alpha.vlen, n);
	}

	clapack_dposv(CblasRowMajor,CblasUpper, n, 1, kernel_matrix.matrix, n,
			m_alpha.vector, n);

	SG_FREE(kernel_matrix.matrix);

	return true;
}

bool CKernelRidgeRegression::train_machine_gs(CFeatures* data)
{
    if (!m_labels)
        SG_ERROR("No labels set\n");

    if (data)
    {
        if (m_labels->get_num_labels() != data->get_num_vectors())
            SG_ERROR("Number of training vectors does not match number of labels\n");
        kernel->init(data, data);
    }
    ASSERT(kernel && kernel->has_features());

    // Get kernel matrix
    SGMatrix<float64_t> A = kernel->get_kernel_matrix<float64_t>();
    int32_t n = A.num_cols;
    int32_t m = A.num_rows;
    ASSERT(A.matrix && m>0 && n>0);

    for(int32_t i=0; i < n; i++)
        A.matrix[i+i*n]+=m_tau;

    // re-set alphas of kernel machine
    m_alpha.destroy_vector();
    SGVector<float64_t> b;
    float64_t alpha_old;

    b=m_labels->get_labels_copy();
    m_alpha=m_labels->get_labels_copy();
    m_alpha.zero();

    for(int32_t i=0; i < n; i++)
    {
        b[i]*=2*m_tau;
    }

    // tell kernel machine that all alphas are needed as 'support vectors'
    m_svs.destroy_vector();
    m_svs=SGVector<index_t>(m_alpha.vlen);
    m_svs.range_fill();

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
                    sigma-=A.matrix[j+i*n]*m_alpha[j];
            alpha_old=m_alpha[i];
            m_alpha[i]=sigma/A.matrix[i+i*n];
            d=fabs(alpha_old-m_alpha[i]);
            if(d>err)
                err=d;
        }
        if (err<=m_epsilon)
            flag=false;
    }

    SG_FREE(A.matrix);

    return true;
}

bool CKernelRidgeRegression::train_machine(CFeatures *data)
{
    return (this->*m_train_func)(data);
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

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *
 * Copyright (C) 2012 Michal Uricar, uricamic@cmp.felk.cvut.cz
 */

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>

#include <shogun/regression/GSKernelRidgeRegression.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CGSKernelRidgeRegression::CGSKernelRidgeRegression()
: CKernelMachine()
{
    init();
}

CGSKernelRidgeRegression::CGSKernelRidgeRegression(float64_t tau, CKernel* k, CLabels* lab)
: CKernelMachine()
{
    init();

    m_tau=tau;
    set_labels(lab);
    set_kernel(k);
    set_epsilon(0.0001);
}

void CGSKernelRidgeRegression::init()
{
    m_tau=1e-6;

    SG_ADD(&m_tau, "tau", "Regularization parameter", MS_AVAILABLE);
}

bool CGSKernelRidgeRegression::train_machine(CFeatures* data)
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

bool CGSKernelRidgeRegression::load(FILE* srcfile)
{
    SG_SET_LOCALE_C;
    SG_RESET_LOCALE;
    return false;
}

bool CGSKernelRidgeRegression::save(FILE* dstfile)
{
    SG_SET_LOCALE_C;
    SG_RESET_LOCALE;
    return false;
}

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

#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CKernelRidgeRegression::CKernelRidgeRegression()
: CKernelMachine()
{
	init();
}

CKernelRidgeRegression::CKernelRidgeRegression(float64_t tau, CKernel* k, CLabels* lab)
: CKernelMachine()
{
	init();

	m_tau=tau;
	set_labels(lab);
	set_kernel(k);
}

void CKernelRidgeRegression::init()
{
	m_tau=1e-6;
	m_epsilon=0.0001;
	SG_ADD(&m_tau, "tau", "Regularization parameter", MS_AVAILABLE);
}

bool CKernelRidgeRegression::solve_krr_system()
{
	SGMatrix<float64_t> kernel_matrix(kernel->get_kernel_matrix());
	int32_t n = kernel_matrix.num_rows;
	SGVector<float64_t> y = ((CRegressionLabels*)m_labels)->get_labels();

	for(index_t i=0; i<n; i++)
		kernel_matrix(i,i) += m_tau;

	Map<MatrixXd> eigen_kernel_matrix(kernel_matrix.matrix, n, n);
	Map<VectorXd> eigen_alphas(m_alpha.vector, n);
	Map<VectorXd> eigen_y(y.vector, n);

	LLT<MatrixXd> llt;
	llt.compute(eigen_kernel_matrix);
	if (llt.info() != Eigen::Success)
	{
		SG_WARNING("Features covariance matrix was not positive definite\n");
		return false;
	}
	eigen_alphas = llt.solve(eigen_y);
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

	if (m_labels->get_num_labels() != kernel->get_num_vec_rhs())
	{
		SG_ERROR("Number of labels does not match number of kernel"
			" columns (num_labels=%d cols=%d\n", m_labels->get_num_labels(), kernel->get_num_vec_rhs());
	}

	// allocate alpha vector
	set_alphas(SGVector<float64_t>(m_labels->get_num_labels()));

	if(!solve_krr_system())
		return false;

	/* tell kernel machine that all alphas are needed as'support vectors' */
	m_svs = SGVector<index_t>(m_alpha.vlen);
	m_svs.range_fill();
	return true;
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

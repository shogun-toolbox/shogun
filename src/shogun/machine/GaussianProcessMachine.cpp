/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/GaussianProcessMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CGaussianProcessMachine::CGaussianProcessMachine()
{
	init();
}

CGaussianProcessMachine::CGaussianProcessMachine(CInferenceMethod* method)
{
	init();
	set_inference_method(method);
}

void CGaussianProcessMachine::init()
{
	m_method=NULL;

	SG_ADD((CSGObject**) &m_method, "inference_method", "Inference method",
	    MS_AVAILABLE);
}

CGaussianProcessMachine::~CGaussianProcessMachine()
{
	SG_UNREF(m_method);
}

SGVector<float64_t> CGaussianProcessMachine::get_posterior_means(CFeatures* data)
{
	REQUIRE(m_method, "Inference method should not be NULL\n")

	CFeatures* feat;

	// use latent features for FITC inference method
	if (m_method->get_inference_type()==INF_FITC)
	{
		CFITCInferenceMethod* fitc_method=
			CFITCInferenceMethod::obtain_from_generic(m_method);
		feat=fitc_method->get_latent_features();
		SG_UNREF(fitc_method);
	}
	else
		feat=m_method->get_features();

	// get kernel and compute kernel matrix: K(feat, data)*scale^2
	CKernel* kernel=m_method->get_kernel();
	kernel->init(feat, data);

	// get kernel matrix and create eigen representation of it
	SGMatrix<float64_t> k_trts=kernel->get_kernel_matrix();
	Map<MatrixXd> eigen_Ks(k_trts.matrix, k_trts.num_rows, k_trts.num_cols);

	// compute Ks=Ks*scale^2
	eigen_Ks*=CMath::sq(m_method->get_scale());

	// cleanup
	SG_UNREF(feat);
	SG_UNREF(kernel);

	// get alpha and create eigen representation of it
	SGVector<float64_t> alpha=m_method->get_alpha();
	Map<VectorXd> eigen_alpha(alpha.vector, alpha.vlen);

	// get mean and create eigen representation of it
	CMeanFunction* mean_function=m_method->get_mean();
	SGVector<float64_t> m=mean_function->get_mean_vector(data);
	Map<VectorXd> eigen_m(m.vector, m.vlen);
	SG_UNREF(mean_function);

	// compute mean: mu=Ks'*alpha+m
	SGVector<float64_t> mu(m.vlen);
	Map<VectorXd> eigen_mu(mu.vector, mu.vlen);
	eigen_mu=eigen_Ks.adjoint()*eigen_alpha+eigen_m;

	return mu;
}

SGVector<float64_t> CGaussianProcessMachine::get_posterior_variances(
		CFeatures* data)
{
	REQUIRE(m_method, "Inference method should not be NULL\n")

	CFeatures* feat;

	// use latent features for FITC inference method
	if (m_method->get_inference_type()==INF_FITC)
	{
		CFITCInferenceMethod* fitc_method=
			CFITCInferenceMethod::obtain_from_generic(m_method);
		feat=fitc_method->get_latent_features();
		SG_UNREF(fitc_method);
	}
	else
		feat=m_method->get_features();

	SG_REF(data);

	// get kernel and compute kernel matrix: K(data, data)*scale^2
	CKernel* kernel=m_method->get_kernel();
	kernel->init(data, data);

	// get kernel matrix and create eigen representation of it
	SGMatrix<float64_t> k_tsts=kernel->get_kernel_matrix();
	Map<MatrixXd> eigen_Kss(k_tsts.matrix, k_tsts.num_rows, k_tsts.num_cols);

	// compute Kss=Kss*scale^2
	eigen_Kss*=CMath::sq(m_method->get_scale());

	// compute kernel matrix: K(feat, data)*scale^2
	kernel->init(feat, data);

	// get kernel matrix and create eigen representation of it
	SGMatrix<float64_t> k_trts=kernel->get_kernel_matrix();
	Map<MatrixXd> eigen_Ks(k_trts.matrix, k_trts.num_rows, k_trts.num_cols);

	// compute Ks=Ks*scale^2
	eigen_Ks*=CMath::sq(m_method->get_scale());

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(feat);
	SG_UNREF(data);

	// get shogun representation of cholesky and create eigen representation
	SGMatrix<float64_t> L=m_method->get_cholesky();
	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);

	// result variance vector
	SGVector<float64_t> s2(k_tsts.num_cols);
	Map<VectorXd> eigen_s2(s2.vector, s2.vlen);

	if (eigen_L.isUpperTriangular())
	{
		// get shogun of diagonal sigma vector and create eigen representation
		SGVector<float64_t> sW=m_method->get_diagonal_vector();
		Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

		// solve L' * V = sW * Ks and compute V.^2
		MatrixXd eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_sW.asDiagonal()*eigen_Ks);
		MatrixXd eigen_sV=eigen_V.cwiseProduct(eigen_V);

		eigen_s2=eigen_Kss.diagonal()-eigen_sV.colwise().sum().adjoint();
	}
	else
	{
		// M = Ks .* (L * Ks)
		MatrixXd eigen_M=eigen_Ks.cwiseProduct(eigen_L*eigen_Ks);
		eigen_s2=eigen_Kss.diagonal()+eigen_M.colwise().sum().adjoint();
	}

	return s2;
}

#endif /* HAVE_EIGEN3 */

/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * Written (W) 2013 Roman Votyakov
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 * Code adapted from
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and 
 * https://gist.github.com/yorkerlin/8a36e8f9b298aa0246a4
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

	// use inducing features for FITC inference method
	if (m_method->get_inference_type()==INF_FITC)
	{
		CFITCInferenceMethod* fitc_method=
			CFITCInferenceMethod::obtain_from_generic(m_method);
		feat=fitc_method->get_inducing_features();
		SG_UNREF(fitc_method);
	}
	else
		feat=m_method->get_features();

	// get kernel and compute kernel matrix: K(feat, data)*scale^2
	CKernel* training_kernel=m_method->get_kernel();
	CKernel* kernel=CKernel::obtain_from_generic(training_kernel->clone());
	SG_UNREF(training_kernel);

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
	SGVector<float64_t> mean=mean_function->get_mean_vector(data);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	SG_UNREF(mean_function);

	const index_t C=alpha.vlen/k_trts.num_rows;
	const index_t n=k_trts.num_rows;
	const index_t m=k_trts.num_cols;

	// compute mean: mu=Ks'*alpha+m
	SGVector<float64_t> mu(C*m);
	Map<MatrixXd> eigen_mu_matrix(mu.vector,C,m);

	for(index_t bl=0; bl<C; bl++)
		eigen_mu_matrix.block(bl,0,1,m)=(eigen_Ks.adjoint()*eigen_alpha.block(bl*n,0,n,1)+eigen_mean).transpose();

	return mu;
}

SGVector<float64_t> CGaussianProcessMachine::get_posterior_variances(
		CFeatures* data)
{
	REQUIRE(m_method, "Inference method should not be NULL\n")

	CFeatures* feat;

	// use inducing features for FITC inference method
	if (m_method->get_inference_type()==INF_FITC)
	{
		CFITCInferenceMethod* fitc_method=
			CFITCInferenceMethod::obtain_from_generic(m_method);
		feat=fitc_method->get_inducing_features();
		SG_UNREF(fitc_method);
	}
	else
		feat=m_method->get_features();

	SG_REF(data);

	// get kernel and compute kernel matrix: K(data, data)*scale^2
	CKernel* training_kernel=m_method->get_kernel();
	CKernel* kernel=CKernel::obtain_from_generic(training_kernel->clone());
	SG_UNREF(training_kernel);

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

	SGVector<float64_t> alpha=m_method->get_alpha();
	const index_t n=k_trts.num_rows;
	const index_t m=k_tsts.num_cols;
	const index_t C=alpha.vlen/n;
	// result variance vector
	SGVector<float64_t> s2(m*C*C);
	Map<VectorXd> eigen_s2(s2.vector, s2.vlen);

	if (eigen_L.isUpperTriangular())
	{
		if (alpha.vlen==L.num_rows)
		{
			//binary case
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
			if (m_method->supports_multiclass())
			{
				//multiclass case
				//see the reference code of the gist link, which is based on the algorithm 3.4 of the GPML textbook
				Map<MatrixXd> &eigen_M=eigen_L;
				eigen_s2.fill(0);

				SGMatrix<float64_t> E=m_method->get_multiclass_E();
				Map<MatrixXd> eigen_E(E.matrix, E.num_rows, E.num_cols);
				ASSERT(E.num_cols==alpha.vlen);
				for(index_t bl_i=0; bl_i<C; bl_i++)
				{
					//n by m
					MatrixXd bi=eigen_E.block(0,bl_i*n,n,n)*eigen_Ks;
					MatrixXd c_cav=eigen_M.triangularView<Upper>().adjoint().solve(bi);
					c_cav=eigen_M.triangularView<Upper>().solve(c_cav);

					for(index_t bl_j=0; bl_j<C; bl_j++)
					{
						MatrixXd bj=eigen_E.block(0,bl_j*n,n,n)*eigen_Ks;
						for (index_t idx_m=0; idx_m<m; idx_m++)
							eigen_s2[bl_j+(bl_i+idx_m*C)*C]=(bj.block(0,idx_m,n,1).array()*c_cav.block(0,idx_m,n,1).array()).sum();
					}
					for (index_t idx_m=0; idx_m<m; idx_m++)
						eigen_s2[bl_i+(bl_i+idx_m*C)*C]+=eigen_Kss(idx_m,idx_m)-(eigen_Ks.block(0,idx_m,n,1).array()*bi.block(0,idx_m,n,1).array()).sum();
				}
			}
			else
			{
				SG_ERROR("Unsupported inference method!\n");
				return s2;
			}
		}
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

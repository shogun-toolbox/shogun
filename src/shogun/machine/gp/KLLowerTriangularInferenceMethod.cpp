 /*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and the reference paper is
 * Challis, Edward, and David Barber.
 * "Concave Gaussian variational approximations for inference in large-scale Bayesian linear models."
 * International conference on Artificial Intelligence and Statistics. 2011.
 *
 * This code specifically adapted from function in approxKL.m and infKL.m
 */

#include <shogun/machine/gp/KLLowerTriangularInferenceMethod.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>

using namespace Eigen;

namespace shogun
{

CKLLowerTriangularInferenceMethod::CKLLowerTriangularInferenceMethod() : CKLInferenceMethod()
{
	init();
}

CKLLowerTriangularInferenceMethod::CKLLowerTriangularInferenceMethod(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CKLInferenceMethod(kern, feat, m, lab, mod)
{
	init();
}

void CKLLowerTriangularInferenceMethod::init()
{
	SG_ADD(&m_InvK_Sigma, "invk_Sigma",
		"K^{-1}Sigma'",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_mean_vec, "mean_vec",
		"The mean vector generated from mean function",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_log_det_Kernel, "log_det_kernel",
		"The Log-determinant of Kernel",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_Kernel_LsD, "L_sqrt_D",
		"The L*sqrt(D) matrix, where L and D are defined in LDLT factorization on Kernel*sq(m_scale)",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_Kernel_P, "Permutation_P",
		"The permutation sequence of P, where P are defined in LDLT factorization on Kernel*sq(m_scale)",
		MS_NOT_AVAILABLE);
	m_log_det_Kernel=0;
}

CKLLowerTriangularInferenceMethod::~CKLLowerTriangularInferenceMethod()
{
}

SGVector<float64_t> CKLLowerTriangularInferenceMethod::get_diagonal_vector()
{
	/** The diagonal vector W is NOT used in this KL method
	 * Therefore, return empty vector
	 */
	return SGVector<float64_t>();
}

void CKLLowerTriangularInferenceMethod::update_deriv()
{
	/** get_derivative_related_cov() does the similar job
	 * Therefore, this function body is empty
	 */
}

void CKLLowerTriangularInferenceMethod::update_init()
{
	Eigen::LDLT<Eigen::MatrixXd> ldlt=update_init_helper();
	MatrixXd Kernel_D=ldlt.vectorD();
	MatrixXd Kernel_L=ldlt.matrixL();
	m_Kernel_LsD=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	m_Kernel_LsD.zero();
	Map<MatrixXd> eigen_Kernel_LsD(m_Kernel_LsD.matrix, m_Kernel_LsD.num_rows, m_Kernel_LsD.num_cols);
	eigen_Kernel_LsD.triangularView<Lower>()=Kernel_L*Kernel_D.array().sqrt().matrix().asDiagonal();
	m_log_det_Kernel=2.0*eigen_Kernel_LsD.diagonal().array().abs().log().sum();

	m_Kernel_P=SGVector<index_t>(m_ktrtr.num_rows);
	for (index_t i=0; i<m_Kernel_P.vlen; i++)
		m_Kernel_P[i]=i;
	Map<VectorXi> eigen_Kernel_P(m_Kernel_P.vector, m_Kernel_P.vlen);
	eigen_Kernel_P=ldlt.transpositionsP()*eigen_Kernel_P;

	m_mean_vec=m_mean->get_mean_vector(m_features);
}

MatrixXd CKLLowerTriangularInferenceMethod::solve_inverse(MatrixXd eigen_A)
{
	Map<VectorXi> eigen_Kernel_P(m_Kernel_P.vector, m_Kernel_P.vlen);
	Map<MatrixXd> eigen_Kernel_LsD(m_Kernel_LsD.matrix, m_Kernel_LsD.num_rows, m_Kernel_LsD.num_cols);

	//re-construct the Permutation Matrix
	PermutationMatrix<Dynamic> P(m_Kernel_P.vlen);
	P.setIdentity();
	SGVector<index_t> tmp=m_Kernel_P.clone();
	for (index_t i=0; i<tmp.vlen; i++)
	{
		while(tmp[i]>i)
		{
			P.applyTranspositionOnTheLeft(i,tmp[i]);
			index_t idx=tmp[i];
			tmp[i]=tmp[idx];
			tmp[idx]=idx;
		}
	}
	P=P.transpose();
	//(P'LDL'P)\eigen_A
	MatrixXd tmp1=P*eigen_A;
	MatrixXd tmp2=eigen_Kernel_LsD.triangularView<Lower>().solve(tmp1);
	MatrixXd tmp3=eigen_Kernel_LsD.triangularView<Lower>().transpose().solve(tmp2);
	return P.transpose()*tmp3;
}

float64_t CKLLowerTriangularInferenceMethod::get_derivative_related_cov(SGMatrix<float64_t> dK)
{
	Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_mu.vlen);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Map<MatrixXd> eigen_InvK_Sigma(m_InvK_Sigma.matrix, m_InvK_Sigma.num_rows, m_InvK_Sigma.num_cols);

	//dnlZ(j)=0.5*sum(sum(dK.*(K\((eye(n)- (invK_V+alpha*m'))')),2),1);
	MatrixXd tmp1=eigen_InvK_Sigma+eigen_alpha*(eigen_mu.transpose());
	MatrixXd tmp2=(MatrixXd::Identity(m_ktrtr.num_rows,m_ktrtr.num_cols)-tmp1).transpose();
	MatrixXd tmp3=solve_inverse(tmp2);
	return 0.5*(tmp3.array()*eigen_dK.array()).sum();
}

void CKLLowerTriangularInferenceMethod::update_approx_cov()
{
	/** update_Sigma() does the similar job
	 * Therefore, this function body is empty
	 */
}

void CKLLowerTriangularInferenceMethod::update_chol()
{
	update_Sigma();
	update_InvK_Sigma();

	m_L=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_InvK_Sigma(m_InvK_Sigma.matrix, m_InvK_Sigma.num_rows, m_InvK_Sigma.num_cols);
	MatrixXd tmp2=(eigen_InvK_Sigma-MatrixXd::Identity(m_ktrtr.num_rows,m_ktrtr.num_cols)).transpose();

	eigen_L=solve_inverse(tmp2);
}

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */

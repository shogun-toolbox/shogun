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
	SG_ADD(&m_noise_factor, "noise_factor",
		"The noise factor used for correcting Kernel matrix",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_exp_factor, "exp_factor",
		"The exponential factor used for increasing noise_factor",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_max_attempt, "max_attempt",
		"The max number of attempt to correct Kernel matrix",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_Kernel_LsD, "L_sqrt_D",
		"The L*sqrt(D) matrix, where L and D are defined in LDLT factorization on Kernel*sq(m_scale)",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_Kernel_P, "Permutation_P",
		"The permutation sequence of P, where P are defined in LDLT factorization on Kernel*sq(m_scale)",
		MS_NOT_AVAILABLE);
	m_log_det_Kernel=0;
	m_noise_factor=1e-16;
	m_max_attempt=10;
	m_exp_factor=2;
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
	/** get_derivative_related_cov(MatrixXd eigen_dK) does the similar job
	 * Therefore, this function body is empty
	 */
}

void CKLLowerTriangularInferenceMethod::update_init()
{
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	eigen_K=eigen_K+m_noise_factor*MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

	Eigen::LDLT<Eigen::MatrixXd> ldlt;
	ldlt.compute(eigen_K*CMath::sq(m_scale));

	float64_t attempt_count=0;
	MatrixXd Kernel_D=ldlt.vectorD();
	while (Kernel_D.minCoeff()<=0)
	{
		if (m_max_attempt>0 && attempt_count>m_max_attempt)
			SG_ERROR("The Kernel matrix is highly non-positive definite",
				" even when adding %f noise to the diagonal elements at max %d attempts\n", m_noise_factor, m_max_attempt);
		attempt_count++;
		float64_t pre_noise_factor=m_noise_factor;
		m_noise_factor*=m_exp_factor;
		//updat the noise  eigen_K=eigen_K+m_noise_factor*(m_exp_factor^attempt_count)*Identity()
		eigen_K=eigen_K+(m_noise_factor-pre_noise_factor)*MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);
		ldlt.compute(eigen_K*CMath::sq(m_scale));
		Kernel_D=ldlt.vectorD();
	}
	MatrixXd Kernel_L=ldlt.matrixL();
	m_Kernel_LsD=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
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
	for (index_t i=0; i<m_Kernel_P.vlen; i++)
	{
		if (m_Kernel_P[i]>i)
			P.applyTranspositionOnTheLeft(i,m_Kernel_P[i]);
	}

	//(P'LDL'P)\eigen_A
	MatrixXd tmp1=P*eigen_A;
	MatrixXd tmp2=eigen_Kernel_LsD.triangularView<Lower>().solve(tmp1);
	MatrixXd tmp3=eigen_Kernel_LsD.triangularView<Lower>().transpose().solve(tmp2);
	return P.transpose()*tmp3;
}

float64_t CKLLowerTriangularInferenceMethod::get_derivative_related_cov(MatrixXd eigen_dK)
{
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

void CKLLowerTriangularInferenceMethod::set_noise_factor(float64_t noise_factor)
{
	REQUIRE(noise_factor>=0, "The noise_factor should be non-negative\n");
	m_noise_factor=noise_factor;
}

void CKLLowerTriangularInferenceMethod::set_max_attempt(index_t max_attempt)
{
	REQUIRE(max_attempt>=0, "The max_attempt should be non-negative. 0 means inifity attempts\n");
	m_max_attempt=max_attempt;
}

void CKLLowerTriangularInferenceMethod::set_exp_factor(float64_t exp_factor)
{
	REQUIRE(exp_factor>1.0, "The exp_factor should be greater than 1.0.\n");
	m_exp_factor=exp_factor;
}


} /* namespace shogun */

#endif /* HAVE_EIGEN3 */

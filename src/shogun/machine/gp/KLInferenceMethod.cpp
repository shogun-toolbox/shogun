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
 * and the reference paper is
 */

#include <shogun/machine/gp/KLInferenceMethod.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/machine/gp/MatrixOperations.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

CKLInferenceMethod::CKLInferenceMethod() : CInferenceMethod()
{
	init();
}

CKLInferenceMethod::CKLInferenceMethod(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
}

void CKLInferenceMethod::init()
{
}

CKLInferenceMethod::~CKLInferenceMethod()
{
}

void CKLInferenceMethod::update()
{
	SG_DEBUG("entering\n");

	CInferenceMethod::update();
	update_alpha();
	update_chol();
	update_approx_cov();
	update_deriv();
	update_parameter_hash();

	SG_DEBUG("leaving\n");
}

SGVector<float64_t> CKLInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(sW);
}

SGVector<float64_t> CKLInferenceMethod::get_alpha()
{
	if (parameter_hash_changed())
		update();

	SGVector<float64_t> result(m_alpha);
	return result;
}

SGMatrix<float64_t> CKLInferenceMethod::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_L);
}

SGVector<float64_t> CKLInferenceMethod::get_posterior_mean()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_mu);
}

SGMatrix<float64_t> CKLInferenceMethod::get_posterior_covariance()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_Sigma);
}

void CKLInferenceMethod::update_approx_cov()
{
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

	m_Sigma=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows,	m_Sigma.num_cols);

	eigen_Sigma = CMatrixOperations::get_inverse(eigen_L, eigen_K, eigen_sW, m_scale);
}

void CKLInferenceMethod::update_chol()
{
	Map<VectorXd> eigen_W(W.vector, W.vlen);

	// create eigen representation of kernel matrix
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	// create shogun and eigen representation of posterior cholesky
	m_L=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

	eigen_L = CMatrixOperations::get_choleksy(eigen_W, eigen_sW, eigen_ktrtr, m_scale);
}

}

#endif /* HAVE_EIGEN3 */

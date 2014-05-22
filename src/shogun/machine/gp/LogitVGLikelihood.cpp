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
 * and the reference paper is
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 *
 * This code specifically adapted from function in approxKL.m
 */

#include <shogun/machine/gp/LogitVGLikelihood.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Integration.h>

using namespace Eigen;

namespace shogun
{

CLogitVGLikelihood::CLogitVGLikelihood()
	: CVariationalGaussianLikelihood()
{
	init();
}

CLogitVGLikelihood::~CLogitVGLikelihood()
{
	SG_UNREF(likelihood);
}

void CLogitVGLikelihood::init()
{
	//Use default Gaussian-Hermite quadrature 20 points
	index_t N = 20;
	m_xgh = SGVector<float64_t>(N);
	m_wgh = SGVector<float64_t>(N);
	CIntegration::generate_gauher(m_xgh, m_wgh);
	likelihood = new CLogitLikelihood();

	SG_ADD(&m_mu, "mu", 
		"The mean of variational normal distribution",
		MS_AVAILABLE, GRADIENT_AVAILABLE);

	SG_ADD(&m_s2, "sigma2", 
		"The variance of variational normal distribution",
		MS_AVAILABLE,GRADIENT_AVAILABLE);

	SG_ADD(&m_lab, "y", 
		"The data/labels (must be -1 or 1) drawn from the distribution",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_log_lam, "log_lam", 
		"The result of used for computing variational expection",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_xgh, "xgh", 
		"Gaussian-Hermite quadrature base points (abscissas)",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_wgh, "wgh", 
		"Gaussian-Hermite quadrature weight factors",
		MS_NOT_AVAILABLE);

}
SGVector<float64_t> CLogitVGLikelihood::get_variational_expection()
{	
	//based on the a_related(m,v,y,lik) function in the Matlab code
	
	//compute expection using numerical integration
	SGVector<float64_t> tmp(m_lab.vlen);
	Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
	Map<MatrixXd> eigen_log_lam(m_log_lam.matrix, m_log_lam.num_rows, m_log_lam.num_cols);

	//a = w'*log_lam;
	Map<VectorXd> eigen_w(m_wgh.vector, m_wgh.vlen);
	eigen_tmp = (eigen_log_lam.array().transpose().colwise() * eigen_w.array()).colwise().sum().matrix();

	return tmp;
}


SGVector<float64_t> CLogitVGLikelihood::get_variational_first_derivative(
		const TParameter* param) const
{
	//based on the a_related(m,v,y,lik) function in the Matlab code
	
	//compute gradient using numerical integration
	REQUIRE(param, "Param is required (param should not be NULL)\n");
	REQUIRE(param->m_name, "Param name is required (param->m_name should not be NULL)\n");
	//We take the derivative wrt to param. Only mu or sigma2 can be the param 
	REQUIRE(!(strcmp(param->m_name, "mu") && strcmp(param->m_name, "sigma2")),
		"Can't compute derivative of the variational expection ", 
		"of log LogitLikelihood using numerical integration ", 
		"wrt %s.%s parameter. The function only accepts mu and sigma2 as parameter",
		get_name(), param->m_name);

	SGVector<float64_t> tmp(m_mu.vlen);
	Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
	Map<VectorXd> eigen_w(m_wgh.vector, m_wgh.vlen);
	Map<VectorXd> eigen_v(m_s2.vector, m_s2.vlen);
	Map<MatrixXd> eigen_log_lam(m_log_lam.matrix, m_log_lam.num_rows, m_log_lam.num_cols);
	Map<VectorXd> eigen_f(m_xgh.vector, m_xgh.vlen);

	if (strcmp(param->m_name, "mu") == 0)
	{
		//Compute the derivative wrt mu

		//f_dm   = f;
		//dm(i)   = (w'*(log_lam.*f_dm  )) /    v(i)^(1/2) ;
		eigen_tmp = (
			(eigen_log_lam.array().transpose().colwise() * eigen_f.array()).colwise() * eigen_w.array()
		).colwise().sum().matrix();
		eigen_tmp = (eigen_tmp.array() / eigen_v.array().sqrt()).matrix();
	}
	else
	{
		//Compute the derivative wrt sigma2

		//f_dV   = f.^2-1;
		//dV(i)   = (w'*(log_lam.*f_dV  )) / (2*v(i)      );
		eigen_tmp = (
			(eigen_log_lam.array().transpose().colwise() * (eigen_f.array().pow(2)-1.0)).colwise() * eigen_w.array()
		).colwise().sum().matrix();
		eigen_tmp = (eigen_tmp.array() / (eigen_v.array()*2.0)).matrix();
	}
	return tmp;
}

void CLogitVGLikelihood::set_variational_distribution(SGVector<float64_t> mu,
	SGVector<float64_t> s2, const CLabels* lab)
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n");

	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
		"Length of the vector of means (%d), length of the vector of "
		"variances (%d) and number of labels (%d) should be the same\n",
		mu.vlen, s2.vlen, lab->get_num_labels());

	REQUIRE(lab->get_label_type()==LT_BINARY,
		"Labels must be type of CBinaryLabels\n");

	for(index_t i = 0; i < s2.vlen; ++i)
		REQUIRE(s2[i] > 0.0, "Variance should always be positive (s2 should be a positive vector)\n");

	m_lab = ((CBinaryLabels*)lab)->get_labels();

	m_mu = mu;
	m_s2 = s2;

	precompute();
}

void CLogitVGLikelihood::precompute()
{
	//samples-by-abscissas
	m_log_lam = SGMatrix<float64_t>(m_s2.vlen, m_xgh.vlen);

	Map<MatrixXd> eigen_log_lam(m_log_lam.matrix, m_log_lam.num_rows, m_log_lam.num_cols);
	Map<VectorXd> eigen_v(m_s2.vector, m_s2.vlen);
	Map<VectorXd> eigen_m(m_mu.vector, m_mu.vlen);
	Map<VectorXd> eigen_y(m_lab.vector, m_lab.vlen);
	Map<VectorXd> eigen_f(m_xgh.vector, m_xgh.vlen);

	//-y(i)*(sqrt(v(i))*f+m(i)) 
	eigen_log_lam = (
		(
			(eigen_f.replicate(1, eigen_v.rows()).array().transpose().colwise() * eigen_v.array().sqrt()).colwise() + eigen_m.array()
		).colwise() * (-eigen_y.array())
	).matrix();

	//-log(1+exp(-yf(ok))); 
	eigen_log_lam = (-(eigen_log_lam.array().exp() + 1.0).log()).matrix();
}

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */

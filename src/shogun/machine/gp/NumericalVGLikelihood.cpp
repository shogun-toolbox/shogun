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
 * and Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 *
 * and the reference paper is
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 *
 * This code specifically adapted from function in infKL.m
 */

#include <shogun/machine/gp/NumericalVGLikelihood.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Integration.h>

using namespace Eigen;

namespace shogun
{

CNumericalVGLikelihood::CNumericalVGLikelihood()
	: CVariationalGaussianLikelihood()
{
	init();
}

CNumericalVGLikelihood::~CNumericalVGLikelihood()
{
}

void CNumericalVGLikelihood::init()
{
	SG_ADD(&m_log_lam, "log_lam", 
		"The result of used for computing variational expection\n",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_xgh, "xgh", 
		"Gaussian-Hermite quadrature base points (abscissas)\n",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_wgh, "wgh", 
		"Gaussian-Hermite quadrature weight factors\n",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_GHQ_N, "GHQ_N", 
		"The number of Gaussian-Hermite quadrature point\n",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_is_init_GHQ, "is_init_GHQ", 
		"Whether Gaussian-Hermite quadrature points are initialized or not\n",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_noise_factor, "noise_factor", 
		"Correct the variance if variance is close to zero or negative\n",
		MS_NOT_AVAILABLE);
	m_GHQ_N=20;
	m_is_init_GHQ=false;
	m_noise_factor=1e-15;
}

void CNumericalVGLikelihood::set_GHQ_number(index_t n)
{
	REQUIRE(n>0, "The number (%d) of Gaussian Hermite point should be positive\n",n);
	if (m_GHQ_N!=n)
	{
		m_GHQ_N=n;
		m_is_init_GHQ=false;
	}
}

SGVector<float64_t> CNumericalVGLikelihood::get_first_derivative_wrt_hyperparameter(
	const TParameter* param) const
{	
	REQUIRE(param, "Param is required (param should not be NULL)\n");
	REQUIRE(param->m_name, "Param name is required (param->m_name should not be NULL)\n");
	if (!(strcmp(param->m_name, "mu") && strcmp(param->m_name, "sigma2")))
		return SGVector<float64_t> ();

	SGVector<float64_t> res(m_lab.vlen);
	res.zero();
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	//ll  = ll  + w(i)*lp;
	CLabels* lab=NULL;

	if (supports_binary())
		lab=new CBinaryLabels(m_lab);
	else if (supports_regression())
		lab=new CRegressionLabels(m_lab);

	for (index_t cidx = 0; cidx < m_log_lam.num_cols; cidx++)
	{
		SGVector<float64_t> tmp(m_log_lam.get_column_vector(cidx), m_log_lam.num_rows, false); 
		SGVector<float64_t> lp=get_first_derivative(lab, tmp, param);
		Map<VectorXd> eigen_lp(lp.vector, lp.vlen);
		eigen_res+=eigen_lp*m_wgh[cidx];
	}

	SG_UNREF(lab);

	return res;
}

SGVector<float64_t> CNumericalVGLikelihood::get_variational_expection()
{	
	SGVector<float64_t> res(m_lab.vlen);
	res.zero();
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	//ll  = ll  + w(i)*lp;
	CLabels* lab=NULL;

	if (supports_binary())
		lab=new CBinaryLabels(m_lab);
	else if (supports_regression())
		lab=new CRegressionLabels(m_lab);

	for (index_t cidx = 0; cidx < m_log_lam.num_cols; cidx++)
	{
		SGVector<float64_t> tmp(m_log_lam.get_column_vector(cidx), m_log_lam.num_rows, false); 
		SGVector<float64_t> lp=get_log_probability_f(lab, tmp);
		Map<VectorXd> eigen_lp(lp.vector, lp.vlen);
		eigen_res+=eigen_lp*m_wgh[cidx];
	}

	SG_UNREF(lab);

	return res;
}

SGVector<float64_t> CNumericalVGLikelihood::get_variational_first_derivative(
		const TParameter* param) const
{
	//based on the likKL(v, lik, varargin) function in infKL.m
	
	//compute gradient using numerical integration
	REQUIRE(param, "Param is required (param should not be NULL)\n");
	REQUIRE(param->m_name, "Param name is required (param->m_name should not be NULL)\n");
	//We take the derivative wrt to param. Only mu or sigma2 can be the param 
	REQUIRE(!(strcmp(param->m_name, "mu") && strcmp(param->m_name, "sigma2")),
		"Can't compute derivative of the variational expection ", 
		"of log LogitLikelihood using numerical integration ", 
		"wrt %s.%s parameter. The function only accepts mu and sigma2 as parameter\n",
		get_name(), param->m_name);

	SGVector<float64_t> res(m_mu.vlen);
	res.zero();
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	Map<VectorXd> eigen_v(m_s2.vector, m_s2.vlen);

	CLabels* lab=NULL;
	
	if (supports_binary())
		lab=new CBinaryLabels(m_lab);
	else if (supports_regression())
		lab=new CRegressionLabels(m_lab);

	if (strcmp(param->m_name, "mu")==0)
	{
		//Compute the derivative wrt mu

		//df  = df  + w(i)*(dlp);
		for (index_t cidx=0; cidx<m_log_lam.num_cols; cidx++)
		{
			SGVector<float64_t> tmp(m_log_lam.get_column_vector(cidx), m_log_lam.num_rows, false); 
			SGVector<float64_t> dlp=get_log_probability_derivative_f(lab, tmp, 1);
			Map<VectorXd> eigen_dlp(dlp.vector, dlp.vlen);
			eigen_res+=eigen_dlp*m_wgh[cidx];
		}
	}
	else
	{
		//Compute the derivative wrt sigma2

		//ai = t(i)./(2*sv+eps); dvi = dlp.*ai; dv = dv + w(i)*dvi;
		VectorXd eigen_sv=eigen_v.array().sqrt().matrix();
		const float64_t EPS=2.2204e-16;

		for (index_t cidx=0; cidx<m_log_lam.num_cols; cidx++)
		{
			SGVector<float64_t> tmp(m_log_lam.get_column_vector(cidx), m_log_lam.num_rows, false); 
			SGVector<float64_t> dlp=get_log_probability_derivative_f(lab, tmp, 1);
			Map<VectorXd> eigen_dlp(dlp.vector, dlp.vlen);
			eigen_res+=((m_wgh[cidx]*0.5*m_xgh[cidx])*eigen_dlp.array()/(eigen_sv.array()+EPS)).matrix();
		}
	}

	SG_UNREF(lab);

	return res;
}

void CNumericalVGLikelihood::set_noise_factor(float64_t noise_factor)
{
	REQUIRE(noise_factor>=0, "The noise_factor (%f) should be non negative\n", noise_factor);
	m_noise_factor=noise_factor;
}

void CNumericalVGLikelihood::set_variational_distribution(SGVector<float64_t> mu,
	SGVector<float64_t> s2, const CLabels* lab)
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n");

	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
		"Length of the vector of means (%d), length of the vector of "
		"variances (%d) and number of labels (%d) should be the same\n",
		mu.vlen, s2.vlen, lab->get_num_labels());

	if (supports_binary())
	{
		REQUIRE(lab->get_label_type()==LT_BINARY,
			"Labels must be type of CBinaryLabels\n");
	}
	else 
	{
		if (supports_regression())
		{
			REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n");
		}
		else
			SG_ERROR("Unsupported Label type\n");
	}

	for(index_t i = 0; i < s2.vlen; ++i)
	{
		REQUIRE(s2[i]+m_noise_factor>0.0,
			"Corrected variational variance (original s2=%f) should always be positive after noise correction (%f)\n",
			s2[i], m_noise_factor);
		if (!(s2[i]>0.0))
			s2[i]+=m_noise_factor;
	}

	if (supports_binary())
		m_lab=((CBinaryLabels*)lab)->get_labels();
	else
		m_lab=((CRegressionLabels*)lab)->get_labels();

	m_mu=mu;
	m_s2=s2;

	if (!m_is_init_GHQ)
	{
		m_xgh=SGVector<float64_t>(m_GHQ_N);
		m_wgh=SGVector<float64_t>(m_GHQ_N);
		CIntegration::generate_gauher(m_xgh, m_wgh);
		m_is_init_GHQ=true;
	}

	precompute();
}

void CNumericalVGLikelihood::precompute()
{
	//samples-by-abscissas
	m_log_lam=SGMatrix<float64_t>(m_s2.vlen, m_xgh.vlen);

	Map<MatrixXd> eigen_log_lam(m_log_lam.matrix, m_log_lam.num_rows, m_log_lam.num_cols);
	Map<VectorXd> eigen_v(m_s2.vector, m_s2.vlen);
	Map<VectorXd> eigen_f(m_mu.vector, m_mu.vlen);
	Map<VectorXd> eigen_t(m_xgh.vector, m_xgh.vlen);

	VectorXd eigen_sv=eigen_v.array().sqrt().matrix();
	//varargin{3} = f + sv*t(i);   % coordinate transform of the quadrature points
	eigen_log_lam = (
			(eigen_t.replicate(1, eigen_v.rows()).array().transpose().colwise()
			 *eigen_sv.array()).colwise()+eigen_f.array()
	).matrix();
}

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */

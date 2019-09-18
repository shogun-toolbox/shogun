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

#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/mathematics/Integration.h>
#endif //USE_GPL_SHOGUN

using namespace Eigen;

namespace shogun
{

NumericalVGLikelihood::NumericalVGLikelihood()
	: VariationalGaussianLikelihood()
{
	init();
}

NumericalVGLikelihood::~NumericalVGLikelihood()
{
}

void NumericalVGLikelihood::init()
{
	SG_ADD(&m_log_lam, "log_lam",
		"The result of used for computing variational expection\n");

	SG_ADD(&m_xgh, "xgh",
		"Gaussian-Hermite quadrature base points (abscissas)\n");

	SG_ADD(&m_wgh, "wgh",
		"Gaussian-Hermite quadrature weight factors\n");

	SG_ADD(&m_GHQ_N, "GHQ_N",
		"The number of Gaussian-Hermite quadrature point\n");

	SG_ADD(&m_is_init_GHQ, "is_init_GHQ",
		"Whether Gaussian-Hermite quadrature points are initialized or not\n");
	m_GHQ_N=20;
	m_is_init_GHQ=false;

}

void NumericalVGLikelihood::set_GHQ_number(index_t n)
{
	REQUIRE(n>0, "The number (%d) of Gaussian Hermite point should be positive\n",n);
	if (m_GHQ_N!=n)
	{
		m_GHQ_N=n;
		m_is_init_GHQ=false;
	}
}

SGVector<float64_t> NumericalVGLikelihood::get_first_derivative_wrt_hyperparameter(
	const std::pair<std::string, std::shared_ptr<const AnyParameter>>& param) const
{
	if (param.first == "mu" && param.first == "sigma2")
		return SGVector<float64_t> ();

	SGVector<float64_t> res(m_lab.vlen);
	res.zero();
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	//ll  = ll  + w(i)*lp;
	std::shared_ptr<Labels> lab=NULL;

	if (supports_binary())
		lab=std::make_shared<BinaryLabels>(m_lab);
	else if (supports_regression())
		lab=std::make_shared<RegressionLabels>(m_lab);

	for (index_t cidx = 0; cidx < m_log_lam.num_cols; cidx++)
	{
		SGVector<float64_t> tmp(m_log_lam.get_column_vector(cidx), m_log_lam.num_rows, false);
		SGVector<float64_t> lp=get_first_derivative(lab, tmp, param);
		Map<VectorXd> eigen_lp(lp.vector, lp.vlen);
		eigen_res+=eigen_lp*m_wgh[cidx];
	}
	return res;
}

SGVector<float64_t> NumericalVGLikelihood::get_variational_expection()
{
	SGVector<float64_t> res(m_lab.vlen);
	res.zero();
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	//ll  = ll  + w(i)*lp;
	std::shared_ptr<Labels> lab=NULL;

	if (supports_binary())
		lab=std::make_shared<BinaryLabels>(m_lab);
	else if (supports_regression())
		lab=std::make_shared<RegressionLabels>(m_lab);

	for (index_t cidx = 0; cidx < m_log_lam.num_cols; cidx++)
	{
		SGVector<float64_t> tmp(m_log_lam.get_column_vector(cidx), m_log_lam.num_rows, false);
		SGVector<float64_t> lp=get_log_probability_f(lab, tmp);
		Map<VectorXd> eigen_lp(lp.vector, lp.vlen);
		eigen_res+=eigen_lp*m_wgh[cidx];
	}



	return res;
}

SGVector<float64_t> NumericalVGLikelihood::get_variational_first_derivative(
		const std::pair<std::string, std::shared_ptr<const AnyParameter>>& param) const
{
	//based on the likKL(v, lik, varargin) function in infKL.m

	//compute gradient using numerical integration
	//We take the derivative wrt to param. Only mu or sigma2 can be the param
	REQUIRE(param.first == "mu" || param.first == "sigma2",
		"Can't compute derivative of the variational expection ",
		"of log LogitLikelihood using numerical integration ",
		"wrt %s.%s parameter. The function only accepts mu and sigma2 as parameter\n",
		get_name(), param.first.c_str());

	SGVector<float64_t> res(m_mu.vlen);
	res.zero();
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	Map<VectorXd> eigen_v(m_s2.vector, m_s2.vlen);

	std::shared_ptr<Labels> lab=NULL;

	if (supports_binary())
		lab=std::make_shared<BinaryLabels>(m_lab);
	else if (supports_regression())
		lab=std::make_shared<RegressionLabels>(m_lab);

	if (param.first == "mu")
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



	return res;
}


bool NumericalVGLikelihood::set_variational_distribution(SGVector<float64_t> mu,
	SGVector<float64_t> s2, std::shared_ptr<const Labels> lab)
{
	bool status = true;
	status = VariationalGaussianLikelihood::set_variational_distribution(mu, s2, lab);

	if (status)
	{
		if (supports_binary())
		{
			REQUIRE(lab->get_label_type()==LT_BINARY,
				"Labels must be type of BinaryLabels\n");
		}
		else
		{
			if (supports_regression())
			{
				REQUIRE(lab->get_label_type()==LT_REGRESSION,
					"Labels must be type of RegressionLabels\n");
			}
			else
				SG_ERROR("Unsupported Label type\n");
		}

		if (supports_binary())
			m_lab=lab->as<BinaryLabels>()->get_labels();
		else
			m_lab=lab->as<RegressionLabels>()->get_labels();

		if (!m_is_init_GHQ)
		{
			m_xgh=SGVector<float64_t>(m_GHQ_N);
			m_wgh=SGVector<float64_t>(m_GHQ_N);
#ifdef USE_GPL_SHOGUN
			Integration::generate_gauher(m_xgh, m_wgh);
#else
			SG_GPL_ONLY
#endif //USE_GPL_SHOGUN
			m_is_init_GHQ=true;
		}

		precompute();

	}

	return status;
}

void NumericalVGLikelihood::precompute()
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


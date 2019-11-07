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
 */

#include <shogun/machine/gp/LaplaceInference.h>


#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

#include <utility>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

LaplaceInference::LaplaceInference() : Inference()
{
	init();
}

LaplaceInference::LaplaceInference(std::shared_ptr<Kernel> kern,
		std::shared_ptr<Features> feat, std::shared_ptr<MeanFunction> m, std::shared_ptr<Labels> lab, std::shared_ptr<LikelihoodModel> mod)
		: Inference(std::move(kern), std::move(feat), std::move(m), std::move(lab), std::move(mod))
{
	init();
}

void LaplaceInference::init()
{
	SG_ADD(&m_dlp, "dlp", "derivative of log likelihood with respect to function location");
	SG_ADD(&m_mu, "mu", "mean vector of the approximation to the posterior");
	SG_ADD(&m_Sigma, "Sigma", "covariance matrix of the approximation to the posterior");
	SG_ADD(&m_W, "W", "the noise matrix");
}

LaplaceInference::~LaplaceInference()
{
}

void LaplaceInference::compute_gradient()
{
	Inference::compute_gradient();

	if (!m_gradient_update)
	{
		update_approx_cov();
		update_deriv();
		m_gradient_update=true;
		update_parameter_hash();
	}
}
void LaplaceInference::update()
{
	SG_TRACE("entering");

	Inference::update();
	update_alpha();
	update_chol();
	m_gradient_update=false;
	update_parameter_hash();

	SG_TRACE("leaving");
}

SGVector<float64_t> LaplaceInference::get_alpha()
{
	if (parameter_hash_changed())
		update();

	SGVector<float64_t> result(m_alpha);
	return result;

}

SGMatrix<float64_t> LaplaceInference::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_L);

}

SGMatrix<float64_t> LaplaceInference::get_posterior_covariance()
{
	compute_gradient();

	return SGMatrix<float64_t>(m_Sigma);
}

}


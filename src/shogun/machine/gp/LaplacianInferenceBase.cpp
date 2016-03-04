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

#include <shogun/machine/gp/LaplacianInferenceBase.h>


#include <shogun/mathematics/Math.h>
#include <shogun/lib/external/brent.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

CLaplacianInferenceBase::CLaplacianInferenceBase() : CInferenceMethod()
{
	init();
}

CLaplacianInferenceBase::CLaplacianInferenceBase(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
}

void CLaplacianInferenceBase::init()
{
	m_iter=20;
	m_tolerance=1e-6;
	m_opt_tolerance=1e-10;
	m_opt_max=10;

	SG_ADD(&m_dlp, "dlp", "derivative of log likelihood with respect to function location", MS_NOT_AVAILABLE);
	SG_ADD(&m_mu, "mu", "mean vector of the approximation to the posterior", MS_NOT_AVAILABLE);
	SG_ADD(&m_Sigma, "Sigma", "covariance matrix of the approximation to the posterior", MS_NOT_AVAILABLE);
	SG_ADD(&m_W, "W", "the noise matrix", MS_NOT_AVAILABLE);
	SG_ADD(&m_tolerance, "tolerance", "amount of tolerance for Newton's iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_iter, "iter", "max Newton's iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_opt_tolerance, "opt_tolerance", "amount of tolerance for Brent's minimization method", MS_NOT_AVAILABLE);
	SG_ADD(&m_opt_max, "opt_max", "max iterations for Brent's minimization method", MS_NOT_AVAILABLE);
}

CLaplacianInferenceBase::~CLaplacianInferenceBase()
{
}

void CLaplacianInferenceBase::compute_gradient()
{
	CInferenceMethod::compute_gradient();

	if (!m_gradient_update)
	{
		update_approx_cov();
		update_deriv();
		m_gradient_update=true;
		update_parameter_hash();
	}
}
void CLaplacianInferenceBase::update()
{
	SG_DEBUG("entering\n");

	CInferenceMethod::update();
	update_alpha();
	update_chol();
	m_gradient_update=false;
	update_parameter_hash();

	SG_DEBUG("leaving\n");
}

SGVector<float64_t> CLaplacianInferenceBase::get_alpha()
{
	if (parameter_hash_changed())
		update();

	SGVector<float64_t> result(m_alpha);
	return result;

}

SGMatrix<float64_t> CLaplacianInferenceBase::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_L);

}

SGMatrix<float64_t> CLaplacianInferenceBase::get_posterior_covariance()
{
	compute_gradient();

	return SGMatrix<float64_t>(m_Sigma);
}

}


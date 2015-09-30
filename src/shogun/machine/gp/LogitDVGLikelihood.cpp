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
 */

#include <shogun/machine/gp/LogitDVGLikelihood.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/machine/gp/LogitVGLikelihood.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>

using namespace Eigen;

namespace shogun
{

CLogitDVGLikelihood::CLogitDVGLikelihood()
	: CDualVariationalGaussianLikelihood()
{
	init();
}

CLogitDVGLikelihood::~CLogitDVGLikelihood()
{
}

SGVector<float64_t> CLogitDVGLikelihood::get_variance_dual_parameter() const
{
	SGVector<float64_t> lambda=m_lambda.clone();

	return lambda;
}

SGVector<float64_t> CLogitDVGLikelihood::get_mu_dual_parameter() const
{
	SGVector<float64_t> alpha=m_lambda.clone();

	for (index_t i=0; i<alpha.vlen; i++)
		alpha[i]-=m_lab[i];

	return alpha;
}

SGVector<float64_t> CLogitDVGLikelihood::get_dual_objective_value()
{
	SGVector<float64_t> result(m_lambda.vlen);

	if (!m_is_valid)
	{
		Map<VectorXd> eigen_reslut(result.vector, result.vlen);
		eigen_reslut.fill(CMath::INFTY);
		return result;
	}
		
	for (index_t i=0; i<result.vlen; ++i)
	{
		float64_t lambda=m_lambda[i];
		result[i]=lambda*CMath::log(lambda)+(1.0-lambda)*CMath::log(1.0-lambda);
	}
	return result;
}

SGVector<float64_t> CLogitDVGLikelihood::get_dual_first_derivative(
		const TParameter* param) const
{
	REQUIRE(param, "Param is required (param should not be NULL)\n");
	REQUIRE(param->m_name, "Param name is required (param->m_name should not be NULL)\n");
	REQUIRE(!strcmp(param->m_name, "lambda"),
		"Can't compute derivative of the variational expection ", 
		"of log LogitLikelihood in dual setting", 
		"wrt %s.%s parameter. The function only accepts lambda as parameter\n",
		get_name(), param->m_name);

	SGVector<float64_t> result(m_lambda.vlen);

	if (!m_is_valid)
	{
		Map<VectorXd> eigen_reslut(result.vector, result.vlen);
		eigen_reslut.fill(CMath::INFTY);
		return result;
	}

	for (index_t i=0; i<result.vlen; ++i)
	{
		float64_t lambda=m_lambda[i];
		result[i]=CMath::log(lambda)-CMath::log(1.0-lambda);
	}
	return result;
}

void CLogitDVGLikelihood::init_likelihood()
{
	set_likelihood(new CLogitVGLikelihood());
}

void CLogitDVGLikelihood::init()
{
	init_likelihood();
}

} /* namespace shogun */
#endif /* HAVE_EIGEN3 */

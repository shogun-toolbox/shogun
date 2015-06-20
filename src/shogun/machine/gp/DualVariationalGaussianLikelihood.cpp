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
#include <shogun/machine/gp/DualVariationalGaussianLikelihood.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>

using namespace Eigen;

namespace shogun
{

CDualVariationalGaussianLikelihood::CDualVariationalGaussianLikelihood()
	: CVariationalGaussianLikelihood()
{
	init();
}

CDualVariationalGaussianLikelihood::~CDualVariationalGaussianLikelihood()
{
}

CVariationalGaussianLikelihood* CDualVariationalGaussianLikelihood::get_variational_likelihood() const
{
	REQUIRE(m_likelihood, "The likelihood model must not be NULL\n");
	CVariationalGaussianLikelihood* var_lik=dynamic_cast<CVariationalGaussianLikelihood *>(m_likelihood);
	REQUIRE(var_lik,
		"The likelihood model (%s) does NOT support variational guassian inference\n",
		m_likelihood->get_name());

	return var_lik;
}

SGVector<float64_t> CDualVariationalGaussianLikelihood::get_variational_expection()
{
	CVariationalLikelihood * var_lik=get_variational_likelihood();
	return var_lik->get_variational_expection();
}

void CDualVariationalGaussianLikelihood::set_noise_factor(float64_t noise_factor)
{
	CVariationalGaussianLikelihood * var_lik=get_variational_likelihood();
	var_lik->set_noise_factor(noise_factor);
}

SGVector<float64_t> CDualVariationalGaussianLikelihood::get_variational_first_derivative(const TParameter* param) const
{
	CVariationalLikelihood * var_lik=get_variational_likelihood();
	return var_lik->get_variational_first_derivative(param);
}

bool CDualVariationalGaussianLikelihood::supports_derivative_wrt_hyperparameter() const
{
	CVariationalLikelihood * var_lik=get_variational_likelihood();
	return var_lik->supports_derivative_wrt_hyperparameter();
}

SGVector<float64_t> CDualVariationalGaussianLikelihood::get_first_derivative_wrt_hyperparameter(const TParameter* param) const
{
	CVariationalLikelihood * var_lik=get_variational_likelihood();
	return var_lik->get_first_derivative_wrt_hyperparameter(param);
}

bool CDualVariationalGaussianLikelihood::set_variational_distribution(
	SGVector<float64_t> mu,	SGVector<float64_t> s2, const CLabels* lab)
{
	CVariationalGaussianLikelihood* var_lik=get_variational_likelihood();
	return var_lik->set_variational_distribution(mu, s2, lab);
}

void CDualVariationalGaussianLikelihood::set_strict_scale(float64_t strict_scale)
{
	REQUIRE((strict_scale>0 && strict_scale<1),
		"The strict_scale (%f) should be between 0 and 1 exclusively.\n",
		strict_scale);
	m_strict_scale=strict_scale;
}

float64_t CDualVariationalGaussianLikelihood::adjust_step_wrt_dual_parameter(SGVector<float64_t> direction, const float64_t step) const
{
	REQUIRE(direction.vlen==m_lambda.vlen,
		"The length (%d) of direction should be same as the length (%d) of dual parameters\n",
		direction.vlen, m_lambda.vlen);

	REQUIRE(step>=0,
		"The step size (%f) should be non-negative\n", step);

	float64_t upper_bound=get_dual_upper_bound();
	float64_t lower_bound=get_dual_lower_bound();

	ASSERT(upper_bound>=lower_bound);

	float64_t min_step=step;

	for (index_t i=0; i<direction.vlen; i++)
	{
		float64_t attempt=m_lambda[i]+step*direction[i];
		float64_t adjust=0;

		if (direction[i]==0.0)
			continue;

		if (lower_bound!=-CMath::INFTY && attempt<lower_bound)
		{
			adjust=(m_lambda[i]-lower_bound)/CMath::abs(direction[i]);
			if (dual_lower_bound_strict())
				adjust*=(1-m_strict_scale);
			if (adjust<min_step)
				min_step=adjust;
		}

		if (upper_bound!=CMath::INFTY && attempt>upper_bound)
		{
			adjust=(upper_bound-m_lambda[i])/CMath::abs(direction[i]);
			if (dual_upper_bound_strict())
				adjust*=(1-m_strict_scale);
			if (adjust<min_step)
				min_step=adjust;
		}
	}

	return min_step;
}

void CDualVariationalGaussianLikelihood::set_dual_parameters(SGVector<float64_t> lambda,  const CLabels* lab)
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n");

	REQUIRE((lambda.vlen==lab->get_num_labels()),
		"Length of the vector of lambda (%d) "
		"and number of labels (%d) should be the same\n",
		lambda.vlen, lab->get_num_labels());
	REQUIRE(lab->get_label_type()==LT_BINARY,
		"Labels (%s) must be type of CBinaryLabels\n",
		lab->get_name());

	m_lab=(((CBinaryLabels*)lab)->get_labels()).clone();

	//Convert the input label to standard label used in the class
	//Note that Shogun uses  -1 and 1 as labels and this class internally uses
	//0 and 1 repectively.
	for(index_t i = 0; i < m_lab.size(); ++i)
		m_lab[i]=CMath::max(m_lab[i], 0.0);

	m_lambda=lambda;

	precompute();
}

bool CDualVariationalGaussianLikelihood::dual_parameters_valid() const
{
	float64_t lower_bound=get_dual_lower_bound();
	float64_t upper_bound=get_dual_upper_bound();

	for (index_t i=0; i<m_lambda.vlen; i++)
	{
		float64_t value=m_lambda[i];
		if (value<lower_bound)
			return false;
		else
		{
			if (dual_lower_bound_strict() && value==lower_bound)
				return false;
			else
			{
				if (value>upper_bound)
					return false;
				else
				{
					if (dual_upper_bound_strict() && value==upper_bound)
						return false;

				}
			}
		}

	}
	return true;
}

void CDualVariationalGaussianLikelihood::precompute()
{
	m_is_valid=dual_parameters_valid();
}

void CDualVariationalGaussianLikelihood::init()
{
	SG_ADD(&m_lambda, "lambda",
		"Dual parameter for variational s2",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_is_valid, "is_valid",
		"Is the Dual parameter valid",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_strict_scale, "strict_scale",
		"The strict variable used in adjust_step_wrt_dual_parameter",
		MS_NOT_AVAILABLE);

	m_is_valid=false;
	m_strict_scale=1e-5;
}

} /* namespace shogun */
#endif /* HAVE_EIGEN3 */

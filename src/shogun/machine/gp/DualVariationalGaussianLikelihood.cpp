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

#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>

using namespace Eigen;

namespace shogun
{

DualVariationalGaussianLikelihood::DualVariationalGaussianLikelihood()
	: VariationalGaussianLikelihood()
{
	init();
}

DualVariationalGaussianLikelihood::~DualVariationalGaussianLikelihood()
{
}

std::shared_ptr<VariationalGaussianLikelihood> DualVariationalGaussianLikelihood::get_variational_likelihood() const
{
	require(m_likelihood, "The likelihood model must not be NULL");
	auto var_lik=std::dynamic_pointer_cast<VariationalGaussianLikelihood>(m_likelihood);
	require(var_lik,
		"The likelihood model ({}) does NOT support variational guassian inference",
		m_likelihood->get_name());

	return var_lik;
}

SGVector<float64_t> DualVariationalGaussianLikelihood::get_variational_expection()
{
	auto var_lik=get_variational_likelihood();
	return var_lik->get_variational_expection();
}

void DualVariationalGaussianLikelihood::set_noise_factor(float64_t noise_factor)
{
	auto var_lik=get_variational_likelihood();
	var_lik->set_noise_factor(noise_factor);
}

SGVector<float64_t> DualVariationalGaussianLikelihood::get_variational_first_derivative(Parameters::const_reference param) const
{
	auto var_lik=get_variational_likelihood();
	return var_lik->get_variational_first_derivative(param);
}

bool DualVariationalGaussianLikelihood::supports_derivative_wrt_hyperparameter() const
{
	auto var_lik=get_variational_likelihood();
	return var_lik->supports_derivative_wrt_hyperparameter();
}

SGVector<float64_t> DualVariationalGaussianLikelihood::get_first_derivative_wrt_hyperparameter(Parameters::const_reference param) const
{
	auto var_lik=get_variational_likelihood();
	return var_lik->get_first_derivative_wrt_hyperparameter(param);
}

bool DualVariationalGaussianLikelihood::set_variational_distribution(
	SGVector<float64_t> mu,	SGVector<float64_t> s2, std::shared_ptr<const Labels> lab)
{
	auto var_lik=get_variational_likelihood();
	return var_lik->set_variational_distribution(mu, s2, lab);
}

void DualVariationalGaussianLikelihood::set_strict_scale(float64_t strict_scale)
{
	require((strict_scale>0 && strict_scale<1),
		"The strict_scale ({}) should be between 0 and 1 exclusively.",
		strict_scale);
	m_strict_scale=strict_scale;
}

float64_t DualVariationalGaussianLikelihood::adjust_step_wrt_dual_parameter(SGVector<float64_t> direction, const float64_t step) const
{
	require(direction.vlen==m_lambda.vlen,
		"The length ({}) of direction should be same as the length ({}) of dual parameters",
		direction.vlen, m_lambda.vlen);

	require(step>=0,
		"The step size ({}) should be non-negative", step);

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

		if (lower_bound!=-Math::INFTY && attempt<lower_bound)
		{
			adjust=(m_lambda[i]-lower_bound)/Math::abs(direction[i]);
			if (dual_lower_bound_strict())
				adjust*=(1-m_strict_scale);
			if (adjust<min_step)
				min_step=adjust;
		}

		if (upper_bound!=Math::INFTY && attempt>upper_bound)
		{
			adjust=(upper_bound-m_lambda[i])/Math::abs(direction[i]);
			if (dual_upper_bound_strict())
				adjust*=(1-m_strict_scale);
			if (adjust<min_step)
				min_step=adjust;
		}
	}

	return min_step;
}

void DualVariationalGaussianLikelihood::set_dual_parameters(SGVector<float64_t> lambda, std::shared_ptr<const Labels> lab)
{
	require(lab, "Labels are required (lab should not be NULL)");

	require((lambda.vlen==lab->get_num_labels()),
		"Length of the vector of lambda ({}) "
		"and number of labels ({}) should be the same",
		lambda.vlen, lab->get_num_labels());
	require(lab->get_label_type()==LT_BINARY,
		"Labels ({}) must be type of BinaryLabels",
		lab->get_name());

	m_lab=lab->as<BinaryLabels>()->get_labels().clone();

	//Convert the input label to standard label used in the class
	//Note that Shogun uses  -1 and 1 as labels and this class internally uses
	//0 and 1 repectively.
	for(index_t i = 0; i < m_lab.size(); ++i)
		m_lab[i]=Math::max(m_lab[i], 0.0);

	m_lambda=lambda;

	precompute();
}

bool DualVariationalGaussianLikelihood::dual_parameters_valid() const
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

void DualVariationalGaussianLikelihood::precompute()
{
	m_is_valid=dual_parameters_valid();
}

void DualVariationalGaussianLikelihood::init()
{
	SG_ADD(&m_lambda, "lambda",
		"Dual parameter for variational s2");

	SG_ADD(&m_is_valid, "is_valid",
		"Is the Dual parameter valid");

	SG_ADD(&m_strict_scale, "strict_scale",
		"The strict variable used in adjust_step_wrt_dual_parameter");

	m_is_valid=false;
	m_strict_scale=1e-5;
}

} /* namespace shogun */

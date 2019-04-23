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

#include <shogun/lib/config.h>
#include <shogun/machine/gp/VariationalLikelihood.h>

namespace shogun
{

VariationalLikelihood::VariationalLikelihood()
	: LikelihoodModel()
{
	init();
}

VariationalLikelihood::~VariationalLikelihood()
{
	
}

void VariationalLikelihood::set_likelihood(std::shared_ptr<LikelihoodModel > lik)
{
	
	m_likelihood=lik;
	
}

void VariationalLikelihood::init()
{
	//m_likelihood will be specified by its subclass
	//via the init_likelihood method
	m_likelihood = NULL;
	

	SG_ADD(&m_lab, "labels",
		"The label of the data\n");

	SG_ADD((std::shared_ptr<SGObject>*)&m_likelihood, "likelihood",
		"The distribution used to model the data\n");
}

SGVector<float64_t> VariationalLikelihood::get_predictive_means(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	std::shared_ptr<const Labels> lab) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_predictive_means(mu, s2, lab);
}

SGVector<float64_t> VariationalLikelihood::get_predictive_variances(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	std::shared_ptr<const Labels> lab) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_predictive_variances(mu, s2, lab);
}

SGVector<float64_t> VariationalLikelihood::get_first_derivative(
	std::shared_ptr<const Labels> lab, SGVector<float64_t> func,
	const TParameter* param) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_first_derivative(lab, func, param);
}

SGVector<float64_t> VariationalLikelihood::get_second_derivative(
	std::shared_ptr<const Labels> lab, SGVector<float64_t> func,
	const TParameter* param) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_second_derivative(lab, func, param);
}

SGVector<float64_t> VariationalLikelihood::get_third_derivative(
	std::shared_ptr<const Labels> lab, SGVector<float64_t> func,
	const TParameter* param) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_third_derivative(lab, func, param);
}

ELikelihoodModelType VariationalLikelihood::get_model_type() const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_model_type();
}

SGVector<float64_t> VariationalLikelihood::get_log_probability_f(
	std::shared_ptr<const Labels> lab, SGVector<float64_t> func) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_log_probability_f(lab, func);
}

SGVector<float64_t> VariationalLikelihood::get_log_probability_derivative_f(
	std::shared_ptr<const Labels> lab, SGVector<float64_t> func, index_t i) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_log_probability_derivative_f(lab, func, i);
}

SGVector<float64_t> VariationalLikelihood::get_log_zeroth_moments(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	std::shared_ptr<const Labels> lab) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_log_zeroth_moments(mu, s2, lab);
}

float64_t VariationalLikelihood::get_first_moment(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	std::shared_ptr<const Labels> lab, index_t i) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_first_moment(mu, s2, lab, i);
}

float64_t VariationalLikelihood::get_second_moment(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	std::shared_ptr<const Labels> lab, index_t i) const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->get_second_moment(mu, s2, lab, i);
}

bool VariationalLikelihood::supports_regression() const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->supports_regression();
}

bool VariationalLikelihood::supports_binary() const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->supports_binary();
}

bool VariationalLikelihood::supports_multiclass() const
{
	require(m_likelihood != NULL, "The likelihood should be initialized");
	return m_likelihood->supports_multiclass();
}

}

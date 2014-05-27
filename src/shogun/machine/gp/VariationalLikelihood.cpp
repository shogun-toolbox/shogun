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

CVariationalLikelihood::CVariationalLikelihood()
	: CLikelihoodModel()
{
	init();
}

CVariationalLikelihood::~CVariationalLikelihood()
{
	SG_UNREF(m_likelihood);
}

void CVariationalLikelihood::set_likelihood(CLikelihoodModel * lik)
{
	SG_UNREF(m_likelihood);
	m_likelihood=lik;
	SG_REF(m_likelihood);
}

void CVariationalLikelihood::init()
{
	//m_likelihood will be specified by its subclass 
	//via the init_likelihood method
	m_likelihood = NULL;
	SG_REF(m_likelihood);

	SG_ADD(&m_lab, "labels", 
		"The label of the data\n",
		MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_likelihood, "likelihood", 
		"The distribution used to model the data\n",
		MS_NOT_AVAILABLE);
}

SGVector<float64_t> CVariationalLikelihood::get_predictive_means(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	const CLabels* lab) const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->get_predictive_means(mu, s2, lab);
}

SGVector<float64_t> CVariationalLikelihood::get_predictive_variances(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	const CLabels* lab) const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->get_predictive_variances(mu, s2, lab);
}

ELikelihoodModelType CVariationalLikelihood::get_model_type() const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->get_model_type();
}

SGVector<float64_t> CVariationalLikelihood::get_log_probability_f(
	const CLabels* lab, SGVector<float64_t> func) const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->get_log_probability_f(lab, func);
}

SGVector<float64_t> CVariationalLikelihood::get_log_probability_derivative_f(
	const CLabels* lab, SGVector<float64_t> func, index_t i) const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->get_log_probability_derivative_f(lab, func, i);
}

SGVector<float64_t> CVariationalLikelihood::get_log_zeroth_moments(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	const CLabels* lab) const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->get_log_zeroth_moments(mu, s2, lab);
}

float64_t CVariationalLikelihood::get_first_moment(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	const CLabels* lab, index_t i) const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->get_first_moment(mu, s2, lab, i);
}

float64_t CVariationalLikelihood::get_second_moment(
	SGVector<float64_t> mu, SGVector<float64_t> s2,
	const CLabels* lab, index_t i) const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->get_second_moment(mu, s2, lab, i);
}

bool CVariationalLikelihood::supports_regression() const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->supports_regression();
}

bool CVariationalLikelihood::supports_binary() const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->supports_binary();
}

bool CVariationalLikelihood::supports_multiclass() const
{
	REQUIRE(m_likelihood != NULL, "The likelihood should be initialized\n");
	return m_likelihood->supports_multiclass();
}

}

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

#ifndef _VARIATIONALGAUSSIANLIKELIHOODMODEL_H_
#define _VARIATIONALGAUSSIANLIKELIHOODMODEL_H_

#include <shogun/lib/config.h>

#include <shogun/machine/gp/VariationalLikelihood.h>
#include <shogun/base/SGObject.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief The variational Gaussian Likelihood base class.
 * The variational distribution is Gaussian
 *
 */
class CVariationalGaussianLikelihood : public CVariationalLikelihood
{
public:
	/** default constructor */
	CVariationalGaussianLikelihood();

	virtual ~CVariationalGaussianLikelihood() {};

	/** set the variational distribution given data and parameters
	 *
	 * @param mu mean of the variational distribution
	 * @param s2 variance of the variational distribution
	 * @param lab labels/data used
	 * @return true if variational parameters are valid
	 *
	 * Note that the variational distribution is Gaussian
	 */
	virtual bool set_variational_distribution(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels* lab);

	/** set a non-negative noise factor in order to correct the variance if variance is close to zero or negative
	 * setting 0 means correction is not applied
	 *
	 * @param noise_factor noise factor
	 *
	 * The default value is 1e-6.
	 */
	virtual void set_noise_factor(float64_t noise_factor);
protected:
	/** The mean of variational Gaussian distribution */
	SGVector<float64_t> m_mu;

	/** The variance of variational Gaussian distribution */
	SGVector<float64_t> m_s2;

	/** this method is called to initialize m_likelihood in init()*/
	virtual void init_likelihood()=0;

private:
	void init();

	/** use to correct the variance if variance is close to zero or negative*/
	float64_t m_noise_factor;
};
}

#endif /* _VARIATIONALGAUSSIANLIKELIHOODMODEL_H_ */

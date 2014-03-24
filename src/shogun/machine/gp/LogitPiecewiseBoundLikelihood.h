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

#ifndef _LOGITPIECEWISEBOUNDLIKELIHOOD_H_
#define _LOGITPIECEWISEBOUNDLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LogitLikelihood.h>

namespace shogun
{

/** @brief Class that models Logit likelihood with variational piecewise bound
 *
 */
class CLogitPiecewiseBoundLikelihood : public CLogitLikelihood
{
public:
	CLogitPiecewiseBoundLikelihood();

	virtual ~CLogitPiecewiseBoundLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name LogitPiecewiseBoundLikelihood
	 */
	virtual const char* get_name() const { return "LogitPiecewiseBoundLikelihood"; }


	/** set the variational piecewise bound for logit likelihood
	 *
	 *  @param bound variational piecewise bound
	 */
	virtual void setbound(const SGMatrix<float64_t> & bound);

	/** get model type
	 *
	 * @return model type Logit
	 */
	virtual ELikelihoodModelType get_model_type() const { return LT_LOGIT; }

	/** returns the expection of the logarithm of a logit distribution 
	 * wrt the variational distribution using piecewise bound
	 *
	 * @param lab labels used
	 * @param mu mean of the variational normal distribution
	 * @param s2 variance of the variational normal distribution
	 *
	 * @return expection
	 */
	virtual SGVector<float64_t> get_variational_expection(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab);

	/** get derivative of the variational expection of log LogitLikelihood
	 * using the piecewise bound with respect to given parameter
	 *
	 * @param lab labels used
	 * @param mu mean of the variational normal distribution
	 * @param s2 variance of the variational normal distribution
	 * @param param parameter
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_variational_first_derivative(const CLabels* lab,
			SGVector<float64_t> mu, SGVector<float64_t> s2, 
			const TParameter* param) const;
private:
	void init();

	SGMatrix<float64_t>  m_bound;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _LOGITPIECEWISEBOUNDLIKELIHOOD_H_ */

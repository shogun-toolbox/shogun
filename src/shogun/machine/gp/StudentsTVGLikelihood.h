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
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and the reference paper is
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 */

#ifndef  _STUDENTSTVGLIKELIHOOD_H_
#define  _STUDENTSTVGLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/machine/gp/NumericalVGLikelihood.h>

namespace shogun
{
template<class C> class SGMatrix;

/** @brief Class that models Student's T likelihood and 
 * uses numerical integration to approximate 
 * the following variational expection of log likelihood
 * \f[
 * \sum_{{i=1}^n}{E_{q(f_i|{\mu}_i,{\sigma}^2_i)}[logP(y_i|f_i)]}
 * \f]
 */
class CStudentsTVGLikelihood : public CNumericalVGLikelihood
{
public:
	CStudentsTVGLikelihood();

	CStudentsTVGLikelihood(float64_t sigma, float64_t df);

	virtual ~CStudentsTVGLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name StudentsTVGLikelihood
	 */
	virtual const char* get_name() const { return "StudentsTVGLikelihood"; }

	/** return whether likelihood function supports
	 * computing the derivative wrt hyperparameter
	 * Note that variational parameters are NOT considered as hyperparameters
	 *
	 * @return boolean
	 */
	virtual bool supports_derivative_wrt_hyperparameter() const { return true; }

protected:
	/** The function used to initialize m_likelihood*/
	virtual void init_likelihood();

private:

	void init();

	/** scale parameter */
	float64_t m_sigma;
	/** degrees of freedom */
	float64_t m_df;

};
}
#endif /* HAVE_EIGEN3 */
#endif /* _STUDENTSTVGLIKELIHOOD_H_ */

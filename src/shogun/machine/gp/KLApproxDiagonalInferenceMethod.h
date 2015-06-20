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
 * and Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and the reference paper is
 * Challis, Edward, and David Barber.
 * "Concave Gaussian variational approximations for inference in large-scale Bayesian linear models."
 * International conference on Artificial Intelligence and Statistics. 2011.
 *
 */

#ifndef _KLAPPROXDIAGONALINFERENCEMETHOD_H_
#define _KLAPPROXDIAGONALINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/machine/gp/KLLowerTriangularInferenceMethod.h>

namespace shogun
{

/** @brief The KL approximation inference method class.
 *
 * The class is implemented based on the KL method in the Challis's paper
 * which uses 1-band (diagonal) represention.
 * Note that in order to do variational inference, each diagonal element should be positive.
 * This implementation updates the diagonal elements in log domain.
 *
 * Code adapted from
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and the reference paper is
 * Challis, Edward, and David Barber.
 * "Concave Gaussian variational approximations for inference in large-scale Bayesian linear models."
 * International conference on Artificial Intelligence and Statistics. 2011.
 *
 * The adapted Matlab code can be found at
 * https://gist.github.com/yorkerlin/d8acb388d03c6976728e
 *
 * Note that "ApproxDiagonal" means a variational diagonal co-variance matrix
 * is used in inference.
 */
class CKLApproxDiagonalInferenceMethod: public CKLLowerTriangularInferenceMethod
{
public:
	/** default constructor */
	CKLApproxDiagonalInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CKLApproxDiagonalInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CKLApproxDiagonalInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name KLApproxDiagonalInferenceMethod
	 */
	virtual const char* get_name() const { return "KLApproxDiagonalInferenceMethod"; }

	/** return what type of inference we are
	 *
	 * @return inference type FITC
	 */
	virtual EInferenceType get_inference_type() const { return INF_KL_DIAGONAL; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CKLApproxDiagonalInferenceMethod object
	 */
	static CKLApproxDiagonalInferenceMethod* obtain_from_generic(CInferenceMethod* inference);

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 */
	virtual SGVector<float64_t> get_alpha();

protected:
	/** update alpha vector */
	virtual void update_alpha();

	/** the helper function to compute
	 * the negative log marginal likelihood
	 *
	 * @return negative log marginal likelihood
	 */
	virtual float64_t get_negative_log_marginal_likelihood_helper();

	/** compute the gradient wrt variational parameters
	 * given the current variational parameters (mu and s2)
	 *
	 * @return gradient of negative log marginal likelihood
	 */
	virtual void get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient);

	/** pre-compute the information for lbfgs optimization.
	 * This function needs to be called before calling
	 * get_negative_log_marginal_likelihood_wrt_parameters()
	 * and/or
	 * get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient)
	 *
	 * @return true if precomputed parameters are valid
	 */
	virtual bool lbfgs_precompute();

	/** compute posterior Sigma matrix*/
	virtual void update_Sigma();

	/** compute inv(corrected_Kernel)*Sigma matrix */
	virtual void update_InvK_Sigma();
private:
	void init();

	SGMatrix<float64_t> m_InvK;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _KLAPPROXDIAGONALINFERENCEMETHOD_H_ */

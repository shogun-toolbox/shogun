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
 * https://gist.github.com/yorkerlin/14ace49f2278f3859614
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and
 * GPstuff - Gaussian process models for Bayesian analysis
 * http://becs.aalto.fi/en/research/bayes/gpstuff/
 *
 * The reference pseudo code is the algorithm 3.3 of the GPML textbook
 *
 */

#ifndef CMULTILAPLACIANINFERENCEMETHOD_H_
#define CMULTILAPLACIANINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/machine/gp/LaplacianInferenceBase.h>

namespace shogun
{

/** @brief The Laplace approximation inference method class for multi classification.
 *
 * This inference method approximates the posterior likelihood function by using
 * Laplace's method. Here, we compute a Gaussian approximation to the posterior
 * via a Taylor expansion around the maximum of the posterior likelihood
 * function.
 *
 * Code adapted from
 * https://gist.github.com/yorkerlin/14ace49f2278f3859614
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and
 * GPstuff - Gaussian process models for Bayesian analysis
 * http://becs.aalto.fi/en/research/bayes/gpstuff/
 *
 * The reference pseudo code is the algorithm 3.3 of the GPML textbook
 */
class CMultiLaplacianInferenceMethod: public CLaplacianInferenceBase
{
public:
	/** default constructor */
	CMultiLaplacianInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CMultiLaplacianInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CMultiLaplacianInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name MultiLaplacian
	 *
	 */
	virtual const char* get_name() const { return "MultiLaplacianInferenceMethod"; }


	/** return what type of inference we are
	 *
	 * @return inference type Laplacian
	 */
	virtual EInferenceType get_inference_type() const { return INF_LAPLACIAN_MULTIPLE; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CMultiLaplacianInferenceMethod object
	 */
	static CMultiLaplacianInferenceMethod* obtain_from_generic(CInferenceMethod* inference);

	/** get negative log marginal likelihood
	 *
	 * @return the negative log of the marginal likelihood function:
	 *
	 * \f[
	 * -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and
	 * \f$\theta\f$ represent hyperparameters.
	 */
	virtual float64_t get_negative_log_marginal_likelihood();

	/** get diagonal vector
	 * where the vector, \f$\pi$\f, defined in the algorithm 3.3 of the GPML textbook
	 *
	 * @return the vector used for inference
	 */
	virtual SGVector<float64_t> get_diagonal_vector();

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports multi classification
	 */
	virtual bool supports_multiclass() const
	{
		check_members();
		return m_model->supports_multiclass();
	}

	/** returns mean vector \f$\mu\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 * Mean vector \f$\mu\f$ is evaluated using Newton's method.
	 *
	 * @return mean vector
	 */
	virtual SGVector<float64_t> get_posterior_mean();

	/** update data all matrices */
	virtual void update();
	
protected:

	/** check if members of object are valid for inference */
	virtual void check_members() const;

	/** update alpha matrix */
	virtual void update_alpha();

	/** update cholesky matrix */
	virtual void update_chol();

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov();

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	virtual void update_deriv();

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInferenceMethod class
	 *
	 * @param param parameter of CInferenceMethod class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inference_method(
			const TParameter* param);

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_likelihood_model(
			const TParameter* param);

	/** returns derivative of negative log marginal likelihood wrt kernel's
	 * parameter
	 *
	 * @param param parameter of given kernel
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_kernel(
			const TParameter* param);

	/** returns derivative of negative log marginal likelihood wrt mean
	 * function's parameter
	 *
	 * @param param parameter of given mean function
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_mean(
			const TParameter* param);
private:

	void init();

protected:

	/** the matrix used to compute gradient wrt hyperparameters */
	SGMatrix<float64_t> m_U;

	/** negative log marginal likelihood */
	float64_t m_nlz;

	/** the helper method used to compute gradient of GP wrt hyperparameter
	 *
	 * @param raw gradient wrt hyperparameter
	 *
	 * @return the gradient of GP wrt hyperparameter
	 *
	 */
	virtual float64_t get_derivative_helper(SGMatrix<float64_t> dK);

	/** the helper used to compute gradient of GP for inference
	 *
	 * construct the \f$\pi$\f vector defined in the algorithm 3.3 of the GPML textbook
	 * Noth that the vector is stored in m_W
	 *
	 */
	virtual void get_dpi_helper();
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CMULTILAPLACIANINFERENCEMETHOD_H_ */

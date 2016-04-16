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

#ifndef CMULTILAPLACEINFERENCEMETHOD_H_
#define CMULTILAPLACEINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#include <shogun/machine/gp/LaplaceInference.h>

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
class CMultiLaplaceInferenceMethod: public CLaplaceInference
{
public:
	/** default constructor */
	CMultiLaplaceInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CMultiLaplaceInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CMultiLaplaceInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name MultiLaplace
	 *
	 */
	virtual const char* get_name() const { return "MultiLaplaceInferenceMethod"; }


	/** return what type of inference we are
	 *
	 * @return inference type Laplace
	 */
	virtual EInferenceType get_inference_type() const { return INF_LAPLACE_MULTIPLE; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CMultiLaplaceInferenceMethod object
	 */
	static CMultiLaplaceInferenceMethod* obtain_from_generic(CInference* inference);

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
	 * where the vector, \f$\pi\f$, defined in the algorithm 3.3 of the GPML textbook
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

	/** get tolerance for newton iterations
	 *
	 * @return tolerance for newton iterations
	 */
	virtual float64_t get_newton_tolerance() { return m_tolerance; }

	/** set tolerance for newton iterations
	 *
	 * @param tol tolerance for newton iterations to set
	 */
	virtual void set_newton_tolerance(float64_t tol) { m_tolerance=tol; }

	/** get max Newton iterations
	 *
	 * @return max Newton iterations
	 */
	virtual int32_t get_newton_iterations() { return m_iter; }

	/** set max Newton iterations
	 *
	 * @param iter max Newton iterations
	 */
	virtual void set_newton_iterations(int32_t iter) { m_iter=iter; }

	/** get tolerance for Brent's minimization method
	 *
	 * @return tolerance for Brent's minimization method
	 */
	virtual float64_t get_minimization_tolerance() { return m_opt_tolerance; }

	/** set tolerance for Brent's minimization method
	 *
	 * @param tol tolerance for Brent's minimization method
	 */
	virtual void set_minimization_tolerance(float64_t tol) { m_opt_tolerance=tol; }

	/** get maximum for Brent's minimization method
	 *
	 * @return maximum for Brent's minimization method
	 */
	virtual float64_t get_minimization_max() { return m_opt_max; }

	/** set maximum for Brent's minimization method
	 *
	 * @param max maximum for Brent's minimization method
	 */
	virtual void set_minimization_max(float64_t max) { m_opt_max=max; }

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
	 * CInference class
	 *
	 * @param param parameter of CInference class
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
	 * @param dK raw gradient wrt hyperparameter
	 *
	 * @return the gradient of GP wrt hyperparameter
	 *
	 */
	virtual float64_t get_derivative_helper(SGMatrix<float64_t> dK);

	/** the helper used to compute gradient of GP for inference
	 *
	 * construct the \f$\pi\f$ vector defined in the algorithm 3.3 of the GPML textbook
	 * Noth that the vector is stored in m_W
	 *
	 */
	virtual void get_dpi_helper();

	/** amount of tolerance for Newton's iterations */
	float64_t m_tolerance;

	/** max Newton's iterations */
	index_t m_iter;

	/** amount of tolerance for Brent's minimization method */
	float64_t m_opt_tolerance;

	/** max iterations for Brent's minimization method */
	float64_t m_opt_max;
};
}
#endif /*  CMULTILAPLACEINFERENCEMETHOD_H_ */

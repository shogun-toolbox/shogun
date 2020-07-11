/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * Written (W) 2013-2014 Heiko Strathmann
 * Written (W) 2013 Roman Votyakov
 * Written (W) 2012 Jacob Walker
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
#ifndef CINFERENCE_H_
#define CINFERENCE_H_

#include <shogun/lib/config.h>


#include <shogun/base/SGObject.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/gp/LikelihoodModel.h>
#include <shogun/machine/gp/MeanFunction.h>
#include <shogun/evaluation/DifferentiableFunction.h>
#include <shogun/optimization/Minimizer.h>

namespace shogun
{

/** inference type */
enum EInferenceType
{
	INF_NONE=0,
	INF_EXACT=10,
	INF_SPARSE=20,
	INF_FITC_REGRESSION=21,
	INF_FITC_LAPLACE_SINGLE=22,
	INF_LAPLACE=30,
	INF_LAPLACE_SINGLE=31,
	INF_LAPLACE_MULTIPLE=32,
	INF_EP=40,
	INF_KL=50,
	INF_KL_DIAGONAL=51,
	INF_KL_CHOLESKY=52,
	INF_KL_COVARIANCE=53,
	INF_KL_DUAL=54,
	INF_KL_SPARSE_REGRESSION=55
};

/** @brief The Inference Method base class.
 *
 * The Inference Method computes (a Gaussian approximation to) the posterior
 * distribution for a given Gaussian Process.
 *
 * It is possible to sample the (true) log-marginal likelihood on the base of
 * any implemented approximation. See
 * CInference::get_marginal_likelihood_estimate.
 */
class Inference : public DifferentiableFunction
{
public:
	/** default constructor */
	Inference();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 */
	Inference(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model);

	~Inference() override;

	/** return what type of inference we are, e.g. exact, FITC, Laplace, etc.
	 *
	 * @return inference type
	 */
	virtual EInferenceType get_inference_type() const { return INF_NONE; }

	/** get negative log marginal likelihood
	 *
	 * @return the negative log of the marginal likelihood function:
	 *
	 * \f[
	 * -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and \f$\theta\f$
	 * represent hyperparameters.
	 */
	virtual float64_t get_negative_log_marginal_likelihood()=0;

	/** Computes an unbiased estimate of the marginal-likelihood (in log-domain),
	 *
	 * \f[
	 * p(y|X,\theta),
	 * \f]
	 * where \f$y\f$ are the labels, \f$X\f$ are the features (omitted from in
	 * the following expressions), and \f$\theta\f$ represent hyperparameters.
	 *
	 * This is done via a Gaussian approximation to the posterior
	 * \f$q(f|y, \theta)\approx p(f|y, \theta)\f$, which is computed by the
	 * underlying CInference instance (if implemented, otherwise error),
	 * and then using an importance sample estimator
	 *
	 * \f[
	 * p(y|\theta)=\int p(y|f)p(f|\theta)df
	 * =\int p(y|f)\frac{p(f|\theta)}{q(f|y, \theta)}q(f|y, \theta)df
	 * \approx\frac{1}{n}\sum_{i=1}^n p(y|f^{(i)})\frac{p(f^{(i)}|\theta)}
	 * {q(f^{(i)}|y, \theta)},
	 * \f]
	 *
	 * where \f$ f^{(i)} \f$ are samples from the posterior approximation
	 * \f$ q(f|y, \theta) \f$. The resulting estimator has a low variance if
	 * \f$ q(f|y, \theta) \f$ is a good approximation. It has large variance
	 * otherwise (while still being consistent). Storing all number of log-domain
	 * ensures numerical stability.
	 *
	 * @param num_importance_samples the number of importance samples \f$n\f$
	 * from \f$ q(f|y, \theta) \f$.
	 * @param ridge_size scalar that is added to the diagonal of the involved
	 * Gaussian distribution's covariance of GP prior and posterior
	 * approximation to stabilise things. Increase if covariance matrix is not
	 * numerically positive semi-definite.
	 *
	 * @return unbiased estimate of the marginal likelihood function
	 * \f$ p(y|\theta),\f$ in log-domain.
	 */
	float64_t get_marginal_likelihood_estimate(int32_t num_importance_samples=1,
			float64_t ridge_size=1e-15);

	/** get log marginal likelihood gradient
	 *
	 * @return vector of the marginal likelihood function gradient with respect
	 * to hyperparameters (under the current approximation to the posterior
	 * \f$q(f|y)\approx p(f|y)\f$:
	 *
	 * \f[
	 * -\frac{\partial log(p(y|X, \theta))}{\partial \theta}
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and \f$\theta\f$
	 * represent hyperparameters.
	 */
	virtual std::map<std::string, SGVector<float64_t>>
	get_negative_log_marginal_likelihood_derivatives(
		std::map<Parameters::value_type, std::shared_ptr<SGObject>> parameters);

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 *
	 * \f[
	 * \mu = K\alpha+meanf
	 * \f]
	 *
	 * where \f$\mu\f$ is the mean,
	 * \f$K\f$ is the prior covariance matrix,
	 * and \f$meanf\f$ is the mean prior fomr MeanFunction
	 *
	 */
	virtual SGVector<float64_t> get_alpha()=0;

	/** get Cholesky decomposition matrix
	 *
	 * @return Cholesky decomposition of matrix
	 *
	 */
	virtual SGMatrix<float64_t> get_cholesky()=0;

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix
	 *
	 */
	virtual SGVector<float64_t> get_diagonal_vector()=0;

	/** returns mean vector \f$\mu\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(\mu,\Sigma)
	 * \f]
	 *
	 * in case if particular inference method doesn't compute posterior
	 * \f$p(f|y)\f$ exactly, and it returns covariance matrix \f$\Sigma\f$ of
	 * the posterior Gaussian distribution \f$\mathcal{N}(\mu,\Sigma)\f$
	 * otherwise.
	 *
	 * @return mean vector
	 */
	virtual SGVector<float64_t> get_posterior_mean()=0;

	/** returns covariance matrix \f$\Sigma\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(\mu,\Sigma)
	 * \f]
	 *
	 * in case if particular inference method doesn't compute posterior
	 * \f$p(f|y)\f$ exactly, and it returns covariance matrix \f$\Sigma\f$ of
	 * the posterior Gaussian distribution \f$\mathcal{N}(\mu,\Sigma)\f$
	 * otherwise.
	 *
	 * @return covariance matrix
	 */
	virtual SGMatrix<float64_t> get_posterior_covariance()=0;

	/** get the gradient
	 *
	 * @param parameters parameter's dictionary
	 *
	 * @return map of gradient. Keys are names of parameters, values are values
	 * of derivative with respect to that parameter.
	 */
	std::map<std::string, SGVector<float64_t>> get_gradient(
		std::map<Parameters::value_type, std::shared_ptr<SGObject>> parameters) override
	{
	        return get_negative_log_marginal_likelihood_derivatives(parameters);
	}

	/** get the function value
	 *
	 * @return vector that represents the function value
	 */
	SGVector<float64_t> get_value() override
	{
		SGVector<float64_t> result(1);
		result[0]=get_negative_log_marginal_likelihood();
		return result;
	}

	/** get features
	*
	* @return features
	*/
	virtual std::shared_ptr<Features> get_features() {  return m_features; }

	/** set features
	*
	* @param feat features to set
	*/
	virtual void set_features(std::shared_ptr<Features> feat)
	{


		m_features=feat;
	}

	/** get kernel
	 *
	 * @return kernel
	 */
	virtual std::shared_ptr<Kernel> get_kernel() {  return m_kernel; }

	/** set kernel
	 *
	 * @param kern kernel to set
	 */
	virtual void set_kernel(std::shared_ptr<Kernel> kern)
	{


		m_kernel=kern;
	}

	/** get mean
	 *
	 * @return mean
	 */
	virtual std::shared_ptr<MeanFunction> get_mean() {  return m_mean; }

	/** set mean
	 *
	 * @param m mean function to set
	 */
	virtual void set_mean(std::shared_ptr<MeanFunction> m)
	{


		m_mean=m;
	}

	/** get labels
	 *
	 * @return labels
	 */
	virtual std::shared_ptr<Labels> get_labels() {  return m_labels; }

	/** set labels
	 *
	 * @param lab label to set
	 */
	virtual void set_labels(std::shared_ptr<Labels> lab)
	{


		m_labels=lab;
	}

	/** get likelihood model
	 *
	 * @return likelihood
	 */
	std::shared_ptr<LikelihoodModel> get_model() {  return m_model; }

	/** set likelihood model
	 *
	 * @param mod model to set
	 */
	virtual void set_model(std::shared_ptr<LikelihoodModel> mod)
	{


		m_model=mod;
	}

	/** get kernel scale
	 *
	 * @return kernel scale
	 */
	virtual float64_t get_scale() const;

	/** set kernel scale
	 *
	 * @param scale scale to be set
	 */
	virtual void set_scale(float64_t scale);

	/** whether combination of inference method and given likelihood function
	 * supports regression
	 *
	 * @return false
	 */
	virtual bool supports_regression() const { return false; }

	/** whether combination of inference method and given likelihood function
	 * supports binary classification
	 *
	 * @return false
	 */
	virtual bool supports_binary() const { return false; }

	/** whether combination of inference method and given likelihood function
	 * supports multiclass classification
	 *
	 * @return false
	 */
	virtual bool supports_multiclass() const { return false; }

	/** update matrices except gradients */
	virtual void update();

	/** get the E matrix used for multi classification
	 *
	 * @return the matrix for multi classification
	 *
	 */
	virtual SGMatrix<float64_t> get_multiclass_E();

	/** Set a minimizer
	 *
	 * @param minimizer minimizer used in inference method
	 */
	virtual void register_minimizer(std::shared_ptr<Minimizer> minimizer);
protected:
	/** check if members of object are valid for inference */
	virtual void check_members() const;

	/** update alpha vector */
	virtual void update_alpha()=0;

	/** update cholesky matrix */
	virtual void update_chol()=0;

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	virtual void update_deriv()=0;

	/** update train kernel matrix */
	virtual void update_train_kernel();

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInference class
	 *
	 * @param param parameter of CInference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inference_method(
			Parameters::const_reference param)=0;

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_likelihood_model(
			Parameters::const_reference param)=0;

	/** returns derivative of negative log marginal likelihood wrt kernel's
	 * parameter
	 *
	 * @param param parameter of given kernel
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_kernel(
			Parameters::const_reference param)=0;

	/** returns derivative of negative log marginal likelihood wrt mean
	 * function's parameter
	 *
	 * @param param parameter of given mean function
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_mean(
			Parameters::const_reference param)=0;

	/** update gradients */
	virtual void compute_gradient();

private:
	void init();

protected:
	/** minimizer */
	std::shared_ptr<Minimizer> m_minimizer;

	/** covariance function */
	std::shared_ptr<Kernel> m_kernel;

	/** mean function */
	std::shared_ptr<MeanFunction> m_mean;

	/** likelihood function to use */
	std::shared_ptr<LikelihoodModel> m_model;

	/** features to use */
	std::shared_ptr<Features> m_features;

	/** labels of features */
	std::shared_ptr<Labels> m_labels;

	/** alpha vector used in process mean calculation */
	SGVector<float64_t> m_alpha;

	/** upper triangular factor of Cholesky decomposition */
	SGMatrix<float64_t> m_L;
	/** kernel scale */
	float64_t m_log_scale = 0.0;

	/** kernel matrix from features (non-scalled by inference scalling) */
	SGMatrix<float64_t> m_ktrtr;

	/** the matrix used for multi classification*/
	SGMatrix<float64_t> m_E;

	/** Whether gradients are updated */
	bool m_gradient_update = false;
};
}
#endif /* CINFERENCE_H_ */

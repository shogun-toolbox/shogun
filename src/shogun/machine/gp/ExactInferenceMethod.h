/*
 * Copyright (c) The Shogun Machine Learning Toolbox
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
#ifndef CEXACTINFERENCEMETHOD_H_
#define CEXACTINFERENCEMETHOD_H_

#include <shogun/lib/config.h>


#include <shogun/machine/gp/Inference.h>

namespace shogun
{

/** @brief The Gaussian exact form inference method class.
 *
 * This inference method computes the Gaussian Method exactly using matrix
 * equations.
 *
 * \f[
 * L = cholesky(K + \sigma^{2}I)
 * \f]
 *
 * \f$L\f$ is the cholesky decomposition of \f$K\f$, the covariance matrix, plus
 * a diagonal matrix with entries \f$\sigma^{2}\f$, the observation noise.
 *
 * \f[
 * \boldsymbol{\alpha} = L^{T} \backslash(L \backslash \boldsymbol{y}})
 * \f]
 *
 * where \f$L\f$ is the matrix mentioned above, \f$\boldsymbol{y}\f$ are the
 * labels, and \f$\backslash\f$ is an operator (\f$x = A \backslash B\f$ means
 * \f$Ax=B\f$.)
 *
 * NOTE: The Gaussian Likelihood Function must be used for this inference
 * method.
 */
class ExactInferenceMethod: public Inference
{
public:
	/** default constructor */
	ExactInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function to use
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 */
	ExactInferenceMethod(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model);

	~ExactInferenceMethod() override;

	/** return what type of inference we are
	 *
	 * @return inference type EXACT
	 */
	EInferenceType get_inference_type() const override { return INF_EXACT; }

	/** returns the name of the inference method
	 *
	 * @return name Exact
	 */
	const char* get_name() const override { return "ExactInferenceMethod"; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted ExactInferenceMethod object
	 */
	static std::shared_ptr<ExactInferenceMethod> obtain_from_generic(const std::shared_ptr<Inference>& inference);

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
	float64_t get_negative_log_marginal_likelihood() override;

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 *
	 * \f[
	 * \mu = K\alpha
	 * \f]
	 *
	 * where \f$\mu\f$ is the mean and \f$K\f$ is the prior covariance matrix.
	 */
	SGVector<float64_t> get_alpha() override;

	/** get Cholesky decomposition matrix
	 *
	 * @return Cholesky decomposition of matrix:
	 *
	 * \f[
	 * L = Cholesky(sW*K*sW+I)
	 * \f]
	 *
	 * where \f$K\f$ is the prior covariance matrix, \f$sW\f$ is the vector
	 * returned by get_diagonal_vector(), and \f$I\f$ is the identity matrix.
	 */
	SGMatrix<float64_t> get_cholesky() override;

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix
	 *
	 * \f[
	 * Cov = (K^{-1}+sW^{2})^{-1}
	 * \f]
	 *
	 * where \f$Cov\f$ is the posterior covariance matrix, \f$K\f$ is the prior
	 * covariance matrix, and \f$sW\f$ is the diagonal vector.
	 */
	SGVector<float64_t> get_diagonal_vector() override;

	/** returns mean vector \f$\mu\f$ of the posterior Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$
	 *
	 * \f[
	 * p(f|y) = \mathcal{N}(\mu,\Sigma)
	 * \f]
	 *
	 * @return mean vector
	 */
	SGVector<float64_t> get_posterior_mean() override;

	/** returns covariance matrix \f$\Sigma\f$ of the posterior Gaussian
	 * distribution \f$\mathcal{N}(\mu,\Sigma)\f$
	 *
	 * \f[
	 * p(f|y) = \mathcal{N}(\mu,\Sigma)
	 * \f]
	 *
	 * @return covariance matrix
	 */
	SGMatrix<float64_t> get_posterior_covariance() override;

	/**
	 * @return whether combination of exact inference method and given
	 * likelihood function supports regression
	 */
	bool supports_regression() const override
	{
		check_members();
		return m_model->supports_regression();
	}

	/** update matrices except gradients*/
	void update() override;

        /** Set a minimizer
         *
         * @param minimizer minimizer used in inference method
         */
	void register_minimizer(std::shared_ptr<Minimizer> minimizer) override;
protected:
	/** check if members of object are valid for inference */
	void check_members() const override;

	/** update alpha matrix */
	void update_alpha() override;

	/** update Cholesky matrix */
	void update_chol() override;

	/** update mean vector of the posterior Gaussian */
	virtual void update_mean();

	/** update covariance matrix of the posterior Gaussian */
	virtual void update_cov();

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	void update_deriv() override;

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInference class
	 *
	 * @param param parameter of CInference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inference_method(
			Parameters::const_reference param) override;

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_likelihood_model(
			Parameters::const_reference param) override;

	/** returns derivative of negative log marginal likelihood wrt kernel's
	 * parameter
	 *
	 * @param param parameter of given kernel
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_kernel(
			Parameters::const_reference param) override;

	/** returns derivative of negative log marginal likelihood wrt mean
	 * function's parameter
	 *
	 * @param param parameter of given mean function
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_mean(
			Parameters::const_reference param) override;

	/** update gradients */
	void compute_gradient() override;
private:
	/** covariance matrix of the the posterior Gaussian distribution */
	SGMatrix<float64_t> m_Sigma;

	/** mean vector of the the posterior Gaussian distribution */
	SGVector<float64_t> m_mu;

	SGMatrix<float64_t> m_Q;
};
}
#endif /* CEXACTINFERENCEMETHOD_H_ */

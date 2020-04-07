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
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 *
 */

#ifndef _KLCOVARIANCEINFERENCEMETHOD_H_
#define _KLCOVARIANCEINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#include <shogun/machine/gp/KLInference.h>

namespace shogun
{

/** @brief The KL approximation inference method class.
 *
 * The class is implemented based on the KL method in the Nickisch's paper
 * Note that lambda (m_W) is a diagonal vector defined in the paper.
 * The implementation apply L-BFGS to finding optimal solution of negative log likelihood.
 * Since lambda is always non-positive according to the paper,
 * this implementation uses log(-lambda) as representation, which assumes lambda is always negative.
 *
 * Code adapted from
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and the reference paper is
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 *
 * The adapted Matlab code can be found at
 * https://gist.github.com/yorkerlin/b64a015491833562d11a
 *
 */
class KLCovarianceInferenceMethod: public KLInference
{
public:
	/** default constructor */
	KLCovarianceInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	KLCovarianceInferenceMethod(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model);

	~KLCovarianceInferenceMethod() override;

	/** returns the name of the inference method
	 *
	 * @return name KLCovarianceInferenceMethod
	 */
	const char* get_name() const override { return "KLCovarianceInferenceMethod"; }

	/** return what type of inference we are
	 *
	 * @return inference type KL_COVARIANCE
	 */
	EInferenceType get_inference_type() const override { return INF_KL_COVARIANCE; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted KLCovarianceInferenceMethod object
	 */
	static std::shared_ptr<KLCovarianceInferenceMethod> obtain_from_generic(const std::shared_ptr<Inference>& inference);

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

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix:
	 *
	 * \f[
	 * Cov = (K^{-1}+sW^{2})^{-1}
	 * \f]
	 *
	 * where \f$Cov\f$ is the posterior covariance matrix, \f$K\f$ is the prior
	 * covariance matrix, and \f$sW\f$ is the diagonal vector.
	 */
	SGVector<float64_t> get_diagonal_vector() override;

protected:
	/** update covariance matrix of the approximation to the posterior */
	void update_approx_cov() override;

	/** update alpha matrix */
	void update_alpha() override;

	/** update cholesky matrix */
	void update_chol() override;

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	void update_deriv() override;

	/** the helper function to compute
	 * the negative log marginal likelihood
	 *
	 * @return negative log marginal likelihood
	 */
	float64_t get_negative_log_marginal_likelihood_helper() override;

	/** compute the gradient wrt variational parameters
	 * given the current variational parameters (mu and s2)
	 *
	 * @return gradient of negative log marginal likelihood
	 */
	void get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient) override;

	/** pre-compute the information for optimization.
	 * This function needs to be called before calling
	 * get_negative_log_marginal_likelihood_wrt_parameters()
	 * and/or
	 * get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient)
	 *
	 * @return true if precomputed parameters are valid
	 */
	bool precompute() override;

	/** compute matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt  hyperparameter in cov function
	 * Note that
	 * get_derivative_wrt_inference_method(Parameters::const_reference param)
	 * and
	 * get_derivative_wrt_kernel(Parameters::const_reference param)
	 * will call this function
	 *
	 * @param dK the gradient wrt hyperparameter related to cov
	 */

	float64_t get_derivative_related_cov(SGMatrix<float64_t> dK) override;
private:
	void init();

	/** square root of noise matrix W */
	SGVector<float64_t> m_sW;

	/** noise matrix */
	SGVector<float64_t> m_W;

	/** the Matrix V, where
	 * L'*V=diag(sW)*K
	 * Note that L' is a lower triangular matrix
	 */
	SGMatrix<float64_t> m_V;

	/** the Matrix A, where
	 * A=I-K*diag(sW)*inv(L)'*inv(L)*diag(sW)
	 */
	SGMatrix<float64_t> m_A;

	/** the gradient of the variational expection wrt sigma2*/
	SGVector<float64_t> m_dv;

	/** the gradient of the variational expection wrt mu*/
	SGVector<float64_t> m_df;

};
}
#endif /* _KLCOVARIANCEINFERENCEMETHOD_H_ */

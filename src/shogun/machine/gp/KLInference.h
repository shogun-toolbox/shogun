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

#ifndef _KLINFERENCE_H_
#define _KLINFERENCE_H_

#include <shogun/lib/config.h>

#include <shogun/machine/gp/Inference.h>
#include <shogun/machine/gp/VariationalGaussianLikelihood.h>

namespace Eigen
{
	template <class, int, int, int, int, int> class Matrix;
	template <class, int> class LDLT;

	typedef Matrix<float64_t,-1,-1,0,-1,-1> MatrixXd;
}

namespace shogun
{
/** @brief The KL approximation inference method class.
 *
 * This inference method approximates the posterior likelihood function by using
 * KL method. Here, we compute a Gaussian approximation to the posterior
 * via minimizing the KL divergence between variational Gaussian distribution
 * and posterior distribution.
 *
 * Code adapted from
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and the reference paper is
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 */
class CKLInference: public CInference
{
friend class KLInferenceCostFunction;
public:
	/** default constructor */
	CKLInference();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CKLInference(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CKLInference();

	/** return what type of inference we are
	 */
	virtual EInferenceType get_inference_type() const { return INF_KL; }

	/** returns the name of the inference method
	 *
	 * @return name KLInference
	 */
	virtual const char* get_name() const { return "KLInference"; }

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

	/** returns mean vector \f$\mu\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 *
	 * @return mean vector
	 */
	virtual SGVector<float64_t> get_posterior_mean();

	/** returns covariance matrix \f$\Sigma=(K^{-1}+W)^{-1}\f$ of the Gaussian
	 * distribution \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to
	 * the posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 * Covariance matrix is evaluated using matrix inversion lemma:
	 *
	 * \f[
	 * (K^{-1}+W)^{-1} = K - KW^{\frac{1}{2}}B^{-1}W^{\frac{1}{2}}K
	 * \f]
	 *
	 * where \f$B=(W^{frac{1}{2}}*K*W^{frac{1}{2}}+I)\f$.
	 *
	 * @return covariance matrix
	 */
	virtual SGMatrix<float64_t> get_posterior_covariance();

	/**
	 * @return whether combination of KL approximation inference method and
	 * given likelihood function supports regression
	 */
	virtual bool supports_regression() const
	{
		check_members();
		return m_model->supports_regression();
	}

	/**
	 * @return whether combination of KL approximation inference method and
	 * given likelihood function supports binary classification
	 */
	virtual bool supports_binary() const
	{
		check_members();
		return m_model->supports_binary();
	}

	/** set variational likelihood model
	 *
	 * @param mod model to set
	 */
	virtual void set_model(CLikelihoodModel* mod);

	/** update all matrices except gradients */
	virtual void update();

	/** get Cholesky decomposition matrix
	 *
	 * @return Cholesky decomposition of matrix:
	 *
	 * \f[
	 * L = cholesky(sW*K*sW+I)
	 * \f]
	 *
	 * where \f$K\f$ is the prior covariance matrix, \f$sW\f$ is the vector
	 * returned by get_diagonal_vector(), and \f$I\f$ is the identity matrix.
	 *
	 * Note that in some sub class L is not the Cholesky decomposition
	 * In this case, L will still be used to compute required matrix for prediction
	 * see CGaussianProcessMachine::get_posterior_variances()
	 */
	virtual SGMatrix<float64_t> get_cholesky();

	/** set noise factor to ensure Kernel matrix to be positive definite
	 * by adding non-negative noise to diagonal elements of Kernel matrix
	 *
	 * @param noise_factor should be non-negative
	 * default value is 1e-10
	 *
	 */
	virtual void set_noise_factor(float64_t noise_factor);

	/** set max attempt to ensure Kernel matrix to be positive definite
	 *
	 * @param max_attempt should be non-negative. 0 means infinity attempts
	 * default value is 0
	 *
	 */
	virtual void set_max_attempt(index_t max_attempt);

	/** set exp factor to exponentially increase noise factor
	 *
	 * @param exp_factor should be greater than 1.0
	 * default value is 2
	 *
	 */
	virtual void set_exp_factor(float64_t exp_factor);

	/** set minimum coeefficient of kernel matrix used in LDLT factorization
	 *
	 * @param min_coeff_kernel should be non-negative
	 * default value is 1e-5
	 *
	 */
	virtual void set_min_coeff_kernel(float64_t min_coeff_kernel);

        /** Set a minimizer
         *
         * @param minimizer minimizer used in inference method
         */
	virtual void register_minimizer(Minimizer* minimizer);
protected:

	/** update gradients */
	virtual void compute_gradient();

	/** The minimum coeefficient of kernel matrix in LDLT factorization used to check whether the kernel matrix is positive definite or not*/
	float64_t m_min_coeff_kernel;

	/** The factor used to ensure kernel matrix to be positive definite */
	float64_t m_noise_factor;

	/** The factor used to exponentially increase noise_factor */
	float64_t m_exp_factor;

	/** Max number of attempt to correct kernel matrix to be positive definite */
	index_t m_max_attempt;

	/** correct the kernel matrix and factorizated the corrected Kernel matrix
	 * for update
	 */
	virtual void update_init();

	/** a helper function used to correct the kernel matrix using LDLT factorization
	 *
	 * @return the LDLT factorization of the corrected kernel matrix
	 */
	virtual Eigen::LDLT<Eigen::MatrixXd,0x1> update_init_helper();

	/** this method is used to dynamic-cast the likelihood model, m_model,
	 * to variational likelihood model.
	 */
	virtual CVariationalGaussianLikelihood* get_variational_likelihood() const;

	/**check the provided likelihood model supports variational inference
	 * @param mod the provided likelihood model
	 *
	 * @return whether the provided likelihood model supports variational inference or not
	 */
	virtual void check_variational_likelihood(CLikelihoodModel* mod) const;

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov()=0;

	/** compute matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt  hyperparameter in cov function
	 * Note that
	 * get_derivative_wrt_inference_method(const TParameter* param)
	 * and
	 * get_derivative_wrt_kernel(const TParameter* param)
	 * will call this function
	 *
	 * @param dK the gradient wrt hyperparameter related to cov
	 */
	virtual float64_t get_derivative_related_cov(SGMatrix<float64_t> dK)=0;

	/** Using an optimizer to estimate posterior parameters */
	virtual float64_t optimization();

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

	/** the helper function to compute
	 * the negative log marginal likelihood
	 *
	 * @return negative log marginal likelihood
	 */
	virtual float64_t get_negative_log_marginal_likelihood_helper()=0;

	/** compute the negative log marginal likelihood
	 * given the current variational parameters (mu and s2)
	 *
	 * @return negative log marginal likelihood
	 */
	virtual float64_t get_nlml_wrt_parameters();

	/** compute the gradient wrt variational parameters
	 * given the current variational parameters (mu and s2)
	 *
	 * @return gradient of negative log marginal likelihood
	 */
	virtual void get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient)=0;

	/** pre-compute the information for optimization.
	 * This function needs to be called before calling
	 * get_negative_log_marginal_likelihood_wrt_parameters()
	 * and/or
	 * get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient)
	 *
	 * @return true if precomputed parameters are valid
	 */
	virtual bool precompute()=0;

	/** mean vector of the approximation to the posterior
	 * Note that m_mu is also a variational parameter
	 */
	SGVector<float64_t> m_mu;

	/** covariance matrix of the approximation to the posterior */
	SGMatrix<float64_t> m_Sigma;

	/** variational parameter sigma2
	 * Note that sigma2 = diag(m_Sigma)
	 */
	SGVector<float64_t> m_s2;
private:
	void init();
};
}
#endif /* _KLINFERENCE_H_ */

/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
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

#ifndef CSINGLEFITCLAPLACEINFERENCEMETHOD_H
#define CSINGLEFITCLAPLACEINFERENCEMETHOD_H


#include <shogun/lib/config.h>
#include <shogun/machine/gp/SingleFITCInference.h>

namespace shogun
{
/** @brief The FITC approximation inference method class
 * for regression and binary Classification.
 * Note that the number of inducing points (m) is usually far less than the number of input points (n).
 * (the time complexity is computed based on the assumption m < n)
 *
 * Warning: the time complexity of method,
 * SingleFITCInference::get_derivative_wrt_kernel(Parameters::const_reference param),
 * depends on the implementation of virtual kernel method,
 * Kernel::get_parameter_gradient_diagonal(param, i).
 * The default time complexity of the kernel method can be O(n^2)
 *
 * Warning: the the time complexity increases from O(m^2*n) to O(n^2*m) if method
 * SingleFITCLaplaceInferenceMethod::get_posterior_covariance() is called
 *
 * This specific implementation was adapted from the infFITC_Laplace.m file in the
 * GPML toolbox.
 */
class SingleFITCLaplaceInferenceMethod: public SingleFITCInference
{
friend class CFITCPsiLine;
friend class SingleFITCLaplaceNewtonOptimizer; 
friend class SingleFITCLaplaceInferenceMethodCostFunction;
public:
	/** default constructor */
	SingleFITCLaplaceInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 * @param inducing_features features to use
	 */
	SingleFITCLaplaceInferenceMethod(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model,
			std::shared_ptr<Features> inducing_features);

	~SingleFITCLaplaceInferenceMethod() override;

	/** returns the name of the inference method
	 *
	 * @return name SingleFITCLaplace
	 */
	const char* get_name() const override { return "SingleFITCLaplaceInferenceMethod"; }


	/** return what type of inference we are
	 *
	 * @return inference type FITC
	 */
	EInferenceType get_inference_type() const override { return INF_FITC_LAPLACE_SINGLE; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted SingleFITCLaplaceInferenceMethod object
	 */
	static std::shared_ptr<SingleFITCLaplaceInferenceMethod> obtain_from_generic(const std::shared_ptr<Inference>& inference);

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports regression
	 */
	bool supports_regression() const override
	{
		check_members();
		return m_model->supports_regression();
	}

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports binary classification
	 */
	bool supports_binary() const override
	{
		check_members();
		return m_model->supports_binary();
	}

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix:
	 *
	 */
	SGVector<float64_t> get_diagonal_vector() override;

	/** returns mean vector \f$\mu\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f_n|y) \approx q(f|y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 * Mean vector \f$\mu\f$ is evaluated using FITC inference and Newton's method.
	 *
	 * @return mean vector
	 */
	SGVector<float64_t> get_posterior_mean() override;

	/** returns covariance matrix \f$\Sigma=(K^{-1}+W)^{-1}\f$ of the Gaussian
	 * distribution \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to
	 * the posterior:
	 *
	 * Note that the time complexity of this method is O(n^3),
	 * where n is the number of samples
	 *
	 * \f[
	 * p(f_n|y) \approx q(f|y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 * @return covariance matrix
	 */
	SGMatrix<float64_t> get_posterior_covariance() override;

	/** update matrices except gradients*/
	void update() override;

	/** get negative log marginal likelihood
	 *
	 * @return the negative log of the (approximated) marginal likelihood function:
	 *
	 * \f[
	 * -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and
	 * \f$\theta\f$ represent hyperparameters.
	 */
	float64_t get_negative_log_marginal_likelihood() override;

	/** Set a minimizer
	 *
	 * @param minimizer minimizer used in inference method
	 */
	void register_minimizer(std::shared_ptr<Minimizer> minimizer) override;

protected:
	/** update gradients */
	void compute_gradient() override;

	/** pre-compution for Newton's method*/
	virtual void update_init();

	/** update alpha matrix */
	void update_alpha() override;

	/** update cholesky matrix */
	void update_chol() override;

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov();

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

	/** efficiently compute the Cholesky decomposition of inverse of the input matrix
	 * chol(inv(mtx))
	 *
	 * @param mtx symmetric positive definite matrix
	 *
	 * @return Cholesky decomposition of inverse of the input matrix
	 */
	virtual SGMatrix<float64_t> get_chol_inv(SGMatrix<float64_t> mtx);

	/** efficiently compute the matrix-vector product
	 * \f$\Sigma \times al\f$, where \f$\Sigma\f$ is
	 * the FITC equivalent covariance n-by-n matrix (prior) of f_n
	 *
	 * @param al input vector
	 *
	 * @return the matrix-vector product
	 */
	virtual SGVector<float64_t> compute_mvmK(SGVector<float64_t> al);

	/** efficiently compute the matrix-vector product
	 * \f$ \inv{\inv{W}+\Sigma} \times x\f$, where \f$\Sigma\f$ is
	 * the FITC equivalent covariance n-by-n matrix (prior) of f_n
	 *
	 * @param x input vector
	 *
	 * @return the matrix-vector product
	 */
	virtual SGVector<float64_t> compute_mvmZ(SGVector<float64_t> x);

	/** returns derivative of negative log marginal likelihood wrt inducing features (input)
	 * Note that in order to call this method, kernel must support FITC inference,
	 * which means derivatives wrt inducing features can be computed
	 *
	 * Note that the kernel must support to compute the derivatives wrt inducing features
	 *
	 * @param param parameter of given kernel
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inducing_features(Parameters::const_reference param) override;

	/** returns derivative of negative log marginal likelihood wrt inducing noise
	 *
	 * @param param parameter of given inference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inducing_noise(Parameters::const_reference param) override;

	/** returns derivative of negative log marginal likelihood wrt param
	 * when W has at least one negative element
	 *
	 * @param res raw derivative
	 * @param param parameter
	 *
	 * @return derivative when W has negative element(s)
	 */
	virtual SGVector<float64_t> derivative_helper_when_Wneg(SGVector<float64_t> res, Parameters::const_reference param);

	/** compute variables which are required to compute negative log marginal
	 * likelihood full derivatives wrt  cov-like hyperparameter \f$\theta\f$
	 *
	 * Note that
	 * scale, which is a hyperparameter in inference_method, is a cov-like hyperparameter
	 * hyperparameters in cov function are cov-like hyperparameters
	 *
	 * @param ddiagKi \f$\textbf{diag}(\frac{\partial {\Sigma_{n}}}{\partial {\theta}})\f$
	 * @param dKuui \f$\frac{\partial {\Sigma_{m}}}{\partial {\theta}}\f$
	 * @param dKui \f$\frac{\partial {\Sigma_{m,n}}}{\partial {\theta}}\f$
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	float64_t get_derivative_related_cov(SGVector<float64_t> ddiagKi,
		SGMatrix<float64_t> dKuui, SGMatrix<float64_t> dKui) override;

	/** helper function to compute variables which are required to compute negative log marginal
	 * likelihood derivatives wrt mean \f$\lambda\f$
	 *
	 * @param dmu \f$\frac{\partial {\mu_{n}}}{\partial {\lambda}}\f$
	 * @return derivative of negative log marginal likelihood
	 */
	float64_t get_derivative_related_mean(SGVector<float64_t> dmu) override;

	/** helper function to compute variables which are required to compute negative log marginal
	 * likelihood implicit derivatives
	 *
	 * @param d raw derivative
	 * @return derivative of negative log marginal likelihood
	 */
	virtual float64_t get_derivative_implicit_term_helper(SGVector<float64_t> d);

	/** compute the function value given the current alpha
	 *
	 * @return the function value
	 */
	float64_t get_psi_wrt_alpha();

	/** compute the gradient given the current alpha
	 *
	 * @param gradient derivative of the function wrt alpha
	 */
	void get_gradient_wrt_alpha(SGVector<float64_t> gradient);

private:
	/** init */
	void init();

protected:

	/** a parameter used to compute function value and gradient for LBFGS update*/
	SGVector<float64_t> m_mean_f;

	/** square root of W */
	SGVector<float64_t> m_sW;

	/** second derivative of log likelihood with respect to function location */
	SGVector<float64_t> m_d2lp;

	/** third derivative of log likelihood with respect to function location */
	SGVector<float64_t> m_d3lp;

	/** derivative of log likelihood with respect to function location */
	SGVector<float64_t> m_dlp;

	/** noise matrix */
	SGVector<float64_t> m_W;

	/** Cholesky of inverse covariance of inducing features */
	SGMatrix<float64_t> m_chol_R0;

	/** derivative of negative log (approximated) marginal likelihood wrt f */
	SGVector<float64_t> m_dfhat;

	/** g defined in infFITC_Laplace.m*/
	SGVector<float64_t> m_g;

	/** the negative log likelihood without constant terms of
	 * \f[
	 * -log(p(f_n|y))
	 * \f]
	 * used in Newton's method
	 * */
	float64_t m_Psi;

	/** d0 defined in infFITC_Laplace.m*/
	SGVector<float64_t> m_dg;

	/** whether W contains negative elements*/
	bool m_Wneg;
};

/** @brief The build-in minimizer for SingleFITCLaplaceInference */
class SingleFITCLaplaceNewtonOptimizer: public Minimizer
{
public:
	SingleFITCLaplaceNewtonOptimizer() :Minimizer() {  init(); }

	const char* get_name() const override { return "SingleFITCLaplaceNewtonOptimizer"; }

	~SingleFITCLaplaceNewtonOptimizer() override {  }

	/** Set the inference method
	 * @param obj the inference method
	 */
	void set_target(const std::shared_ptr<SingleFITCLaplaceInferenceMethod >&obj);

	/** Unset the inference method
	 * @param is_unref do we SG_UNREF the method
	 */
	void unset_target(bool is_unref);
	
	/** Do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	float64_t minimize() override;

	/** set maximum for Brent's minimization method
	 *
	 * @param max maximum for Brent's minimization method
	 */
	virtual void set_minimization_max(float64_t max) { m_opt_max=max; }

	/** set tolerance for Brent's minimization method
	 *
	 * @param tol tolerance for Brent's minimization method
	 */
	virtual void set_minimization_tolerance(float64_t tol) { m_opt_tolerance=tol; }

	/** set max Newton iterations
	 *
	 * @param iter max Newton iterations
	 */
	virtual void set_newton_iterations(int32_t iter) { m_iter=iter; }

	/** set tolerance for newton iterations
	 *
	 * @param tol tolerance for newton iterations to set
	 */
	virtual void set_newton_tolerance(float64_t tol) { m_tolerance=tol; }
private:
	void init();

	/** the inference method */
	std::shared_ptr<SingleFITCLaplaceInferenceMethod >m_obj;

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
#endif /* CSINGLEFITCLAPLACEINFERENCEMETHOD_H */

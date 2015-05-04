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

#ifndef CSINGLEFITCLAPLACIANINFERENCEMETHOD_H
#define CSINGLEFITCLAPLACIANINFERENCEMETHOD_H

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/SingleFITCLaplacianBase.h>

namespace shogun
{

/** @brief The SingleFITCLaplace approximation inference method class
 * for regression and binary Classification.
 * Note that the number of inducing points (m) is usually far less than the number of input points (n).
 * (the time complexity is computed based on the assumption m < n)
 *
 * Warning: the time complexity of method,
 * CSingleFITCLaplacianBase::get_derivative_wrt_kernel(const TParameter* param),
 * depends on the implementation of virtual kernel method,
 * CKernel::get_parameter_gradient_diagonal(param, i).
 * The default time complexity of the kernel method can be O(n^2)
 *
 * Warning: the the time complexity increases from O(m^2*n) to O(n^2*m) if method
 * CSingleFITCLaplacianInferenceMethod::get_posterior_covariance() is called
 *
 * This specific implementation was adapted from the infFITC_Laplace.m file in the
 * GPML toolbox.
 */
class CSingleFITCLaplacianInferenceMethod: public CSingleFITCLaplacianBase
{
friend class CFITCPsiLine;
public:
	/** default constructor */
	CSingleFITCLaplacianInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 * @param inducing_features features to use
	 */
	CSingleFITCLaplacianInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model,
			CFeatures* inducing_features);

	virtual ~CSingleFITCLaplacianInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name SingleFITCLaplacian
	 */
	virtual const char* get_name() const { return "SingleFITCLaplacianInferenceMethod"; }


	/** return what type of inference we are
	 *
	 * @return inference type FITC
	 */
	virtual EInferenceType get_inference_type() const { return INF_FITC_LAPLACIAN_SINGLE; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CSingleFITCLaplacianInferenceMethod object
	 */
	static CSingleFITCLaplacianInferenceMethod* obtain_from_generic(CInferenceMethod* inference);

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports regression
	 */
	virtual bool supports_regression() const
	{
		check_members();
		return m_model->supports_regression();
	}

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports binary classification
	 */
	virtual bool supports_binary() const
	{
		check_members();
		return m_model->supports_binary();
	}

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix:
	 *
	 */
	virtual SGVector<float64_t> get_diagonal_vector();

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
	virtual SGVector<float64_t> get_posterior_mean();

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
	virtual SGMatrix<float64_t> get_posterior_covariance();

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

	/** update data all matrices */
	virtual void update();

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
	virtual float64_t get_negative_log_marginal_likelihood();
protected:
	/** pre-compution for Newton's method*/
	virtual void update_init();

	/** update alpha matrix */
	virtual void update_alpha();

	/** update cholesky matrix */
	virtual void update_chol();

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

	/** efficiently compute the Cholesky decomposition of inverse of the input matrix
	 * chol(inv(mtx))
	 *
	 * @param mtx symmetric positive definite matrix
	 *
	 * @return Cholesky decomposition of inverse of the input matrix
	 */
	virtual SGMatrix<float64_t> get_chol_inv(SGMatrix<float64_t> mtx);

	/** efficiently compute the matrix-vector product
	 * \f$\Sigma \times al$\f, where \f$\Sigma$\f is
	 * the FITC equivalent covariance n-by-n matrix (prior) of f_n
	 *
	 * @param al input vector
	 *
	 * @return the matrix-vector product
	 */
	virtual SGVector<float64_t> compute_mvmK(SGVector<float64_t> al);

	/** efficiently compute the matrix-vector product
	 * \f$ \inv{\inv{W}+\Sigma} \times x$\f, where \f$\Sigma$\f is
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
	virtual SGVector<float64_t> get_derivative_wrt_inducing_features(const TParameter* param);

	/** returns derivative of negative log marginal likelihood wrt inducing noise
	 *
	 * @param param parameter of given inference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inducing_noise(const TParameter* param);

	/** returns derivative of negative log marginal likelihood wrt param
	 * when W has at least one negative element
	 *
	 * @param res raw derivative
	 * @param param parameter
	 *
	 * @return derivative when W has negative element(s)
	 */
	virtual SGVector<float64_t> derivative_helper_when_Wneg(SGVector<float64_t> res, const TParameter* param);

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
	virtual float64_t get_derivative_related_cov(SGVector<float64_t> ddiagKi,
		SGMatrix<float64_t> dKuui, SGMatrix<float64_t> dKui);

	/** helper function to compute variables which are required to compute negative log marginal
	 * likelihood derivatives wrt mean \f$\lambda\f$
	 *
	 * @param dmu \f$\frac{\partial {\mu_{n}}}{\partial {\lambda}}\f$
	 * @return derivative of negative log marginal likelihood
	 */
	virtual float64_t get_derivative_related_mean(SGVector<float64_t> dmu);

	/** helper function to compute variables which are required to compute negative log marginal
	 * likelihood implicit derivatives
	 *
	 * @param d raw derivative
	 * @return derivative of negative log marginal likelihood
	 */
	virtual float64_t get_derivative_implicit_term_helper(SGVector<float64_t> d);
private:
	/** init */
	void init();

protected:
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

	/** amount of tolerance for Newton's iterations */
	float64_t m_tolerance;

	/** max Newton's iterations */
	index_t m_iter;

	/** amount of tolerance for Brent's minimization method */
	float64_t m_opt_tolerance;

	/** max iterations for Brent's minimization method */
	float64_t m_opt_max;

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
}
#endif /* HAVE_EIGEN3 */
#endif /* CSINGLEFITCLAPLACIANINFERENCEMETHOD_H */

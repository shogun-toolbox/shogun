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

#ifndef CSINGLEFITCLAPLACIANBASE_H
#define CSINGLEFITCLAPLACIANBASE_H

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/FITCInferenceBase.h>
#include <shogun/lib/Lock.h>

namespace shogun
{

/** @brief The Fully Independent Conditional Training inference base class
 * for Laplace and regression for 1-D labels (1D regression and binary classification)
 *
 * This base class implements the (explicit) derivatives of negative log marginal likelihood
 * wrt hyperparameter for FITC regression and FITC single Laplace.
 * For FITC single Laplace, we can compute further implicit derivatives.
 * For FITC regression, these explicit derivatives are the full derivatives.
 *
 * For more details, see Quiñonero-Candela, Joaquin, and Carl Edward Rasmussen.
 * "A unifying view of sparse approximate Gaussian process regression."
 * The Journal of Machine Learning Research 6 (2005): 1939-1959.
 *
 * Note that the number of inducing points (m) is usually far less than the number of input points (n).
 * (the time complexity is computed based on the assumption m < n)
 *
 * This specific implementation was inspired by the infFITC.m and infFITC_Laplace.m file
 * in the GPML toolbox.
 *
 * Warning: the time complexity of method,
 * CSingleFITCLaplacianBase::get_derivative_wrt_kernel(const TParameter* param),
 * depends on the implementation of virtual kernel method,
 * CKernel::get_parameter_gradient_diagonal(param, i).
 * The default time complexity of the kernel method can be O(n^2)
 *
 */
class CSingleFITCLaplacianBase: public CFITCInferenceBase
{
public:
	/** default constructor */
	CSingleFITCLaplacianBase();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 * @param inducing_features features to use
	 */
	CSingleFITCLaplacianBase(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model,
			CFeatures* inducing_features);

	virtual ~CSingleFITCLaplacianBase();

	/** returns the name of the inference method
	 *
	 * @return name SingleFITCLaplacianBase
	 */
	virtual const char* get_name() const { return "SingleFITCLaplacianBase"; }

	/** set kernel
	 *
	 * @param kern kernel to set
	 */
	virtual void set_kernel(CKernel* kern);

#ifdef HAVE_NLOPT
	/** opitmize inducing features
	 */
	virtual void optimize_inducing_features();

	/** set the lower bound of inducing features
	 *
	 * @param bound lower bound constrains of inducing features
	 *
	 * Note that if the length of the bound can be 1,
	 * it means the bound constraint applies to each dimension of inducing features
	 *
	 * Note that if the length of the bound is greater than 1,
	 * it means each dimension of the bound constraint applies to the corresponding dimension of inducing features
	 */
	virtual void set_lower_bound_of_inducing_features(SGVector<float64_t> bound);

	/** set the upper bound of inducing features
	 *
	 * @param bound upper bound constrains of inducing features
	 *
	 * Note that if the length of the bound can be 1,
	 * it means the bound constraint applies to each dimension of inducing features
	 *
	 * Note that if the length of the bound is greater than 1,
	 * it means each dimension of the bound constraint applies to the corresponding dimension of inducing features
	 */
	virtual void set_upper_bound_of_inducing_features(SGVector<float64_t> bound);

	/** set the tolearance used in optimization of inducing features
	 *
	 * @param tol tolearance 
	 */
	virtual void set_tolearance_for_inducing_features(float64_t tol);

	/** set the max number of iterations used in optimization of inducing features
	 *
	 * @param it max number of iterations
	 */
	virtual void set_max_iterations_for_inducing_features(int32_t it);

	/** whether enable to opitmize inducing features
	 *
	 * @param is_optmization enable optimization
	 */
	virtual void enable_optimizing_inducing_features(bool is_optmization);
#endif

protected:

#ifdef HAVE_NLOPT
	/** check the bound constraint is vailid or not
	 *
	 * @param bound bound constrains of inducing features
	 */
	virtual void check_bound(SGVector<float64_t> bound);

	/** lower bound of inducing features */
	SGVector<float64_t> m_lower_bound;

	/** upper bound of inducing features */
	SGVector<float64_t> m_upper_bound;

	/**  max number of iterations */
	float64_t m_max_ind_iterations;

	/**  tolearance used in optimizing inducing_features */
	float64_t m_ind_tolerance;

	/**  whether optimize inducing features */
	bool m_opt_inducing_features;
#endif

	/** check whether the provided kernel can
	 * compute the gradient wrt inducing features
	 *
	 * Note that currently we check the name of the provided kernel
	 * to determine whether the kernel can compute the derivatives wrt inducing_features
	 *
	 * The name of a supported Kernel must end with "FITCKernel"
	 */
	virtual void check_fully_FITC();

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

	/** compute variables which are required to compute negative log marginal
	 * likelihood derivatives wrt cov-like hyperparameter \f$\theta\f$
	 *
	 * Note that
	 * scale, which is a hyperparameter in inference_method, is a cov-like hyperparameter
	 * hyperparameters in cov function are cov-like hyperparameters
	 *
	 * @param ddiagKi \f$\textbf{diag}(\frac{\partial {\Sigma_{n}}}{\partial {\theta}})\f$
	 * @param dKuui \f$\frac{\partial {\Sigma_{m}}}{\partial {\theta}}\f$
	 * @param dKui \f$\frac{\partial {\Sigma_{m,n}}}{\partial {\theta}}\f$
	 * @param v auxiliary variable related to explicit derivative
	 * @param R auxiliary variable related to explicit derivative
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual float64_t get_derivative_related_cov(SGVector<float64_t> ddiagKi,
		SGMatrix<float64_t> dKuui, SGMatrix<float64_t> dKui,
		SGVector<float64_t> v, SGMatrix<float64_t> R);

	/** helper function to compute variables which are required to compute negative log marginal
	 * likelihood derivatives wrt cov-like hyperparameter \f$\theta\f$
	 *
	 * Note that
	 * scale, which is a hyperparameter in inference_method, is a cov-like hyperparameter
	 * hyperparameters in cov function are cov-like hyperparameters
	 * what is more, derivative wrt inducing_noise will also use this function
	 *
	 * @param dKuui \f$\frac{\partial {\Sigma_{m}}}{\partial {\theta}}\f$
	 * @param v auxiliary variable related to explicit derivative
	 * @param R auxiliary variable related to explicit derivative
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual float64_t get_derivative_related_cov_helper(SGMatrix<float64_t> dKuui,
		SGVector<float64_t> v, SGMatrix<float64_t> R);

	/** returns derivative of negative log marginal likelihood wrt inducing noise
	 *
	 * @param param parameter of given inference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inducing_noise(
		const TParameter* param);

	/** helper function to compute variables which are required to compute negative log marginal
	 * likelihood derivatives wrt the diagonal part of cov-like hyperparameter \f$\theta\f$
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_related_cov_diagonal();

	/** helper function to compute variables which are required to compute negative log marginal
	 * likelihood derivatives wrt mean \f$\lambda\f$
	 *
	 * @param dmu \f$\frac{\partial {\mu_{n}}}{\partial {\lambda}}\f$
	 * @return derivative of negative log marginal likelihood
	 */
	virtual float64_t get_derivative_related_mean(SGVector<float64_t> dmu);

	/** helper function to compute variables which are required to compute negative log marginal
	 * likelihood derivatives wrt inducing features
	 *
	 * Note that the kernel must support to compute the derivatives wrt inducing features
	 *
	 * @param BdK auxiliary variable related to explicit derivative or implicit derivative
	 * @param param parameter of given kernel
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_related_inducing_features(
	SGMatrix<float64_t> BdK, const TParameter* param);

	/** update alpha vector */
	virtual void update_alpha()=0;

	/** update cholesky matrix */
	virtual void update_chol()=0;

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	virtual void update_deriv()=0;

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInferenceMethod class
	 *
	 * @param param parameter of given inference class
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
			const TParameter* param)=0;

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

	/** Note that alpha is NOT post.alpha
	 * alpha and post.alpha are defined in infFITC.m and infFITC_Laplace.m
	 * */
	SGVector<float64_t> m_al;

	/** t=1/g_sn2 in regression, where g_sn2 is defined in infFITC.m
	 * t=W.*dd in Laplace for binary classification,
	 * where W and dd are defined in infFITC_Laplace.m
	 * */
	SGVector<float64_t> m_t;

	/* B is defined in infFITC.m and infFITC_Laplace.m */
	SGMatrix<float64_t> m_B;

	/* w=B*al */
	SGVector<float64_t> m_w;

	/** Rvdd=W
	 * where W is defined in infFITC.m and Rvdd is defined in infFITC_Laplace.m
	 * Note that W is NOT the diagonal matrix
	 */
	SGMatrix<float64_t> m_Rvdd;

	/* a lock used to parallelly compute derivatives wrt hyperparameters */
	CLock* m_lock;

	/** whether the kernel supports
	 * to compute the derivatives wrt to inducing features
	 */
	bool m_fully_FITC;

	/* V defined in infFITC.m and infFITC_Laplace.m */
	SGMatrix<float64_t> m_V;

private:
	/* init */
	void init();

	/** helper function is passed to the nlopt API
	 *
	 * @param n the length of the variables to be optimized (minimized)
	 * @param x pointer of the variables
	 * @param grad pointer of gradients of current x to be stored
	 * @param func_data pointer of extra information used in opitmization
	 * @return negative marginal log likelihood (minimized function value)
	 * */
	static double nlopt_function(unsigned n, const double* x, double* grad, void* func_data);
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CSINGLEFITCLAPLACIANBASE_H */

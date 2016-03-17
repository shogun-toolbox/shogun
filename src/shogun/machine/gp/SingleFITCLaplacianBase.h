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


#include <shogun/machine/gp/SingleSparseInferenceBase.h>
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
class CSingleFITCLaplacianBase: public CSingleSparseInferenceBase
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

protected:

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
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_likelihood_model(
			const TParameter* param)=0;

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

	/** B is defined in infFITC.m and infFITC_Laplace.m */
	SGMatrix<float64_t> m_B;

	/** w=B*al */
	SGVector<float64_t> m_w;

	/** Rvdd=W
	 * where W is defined in infFITC.m and Rvdd is defined in infFITC_Laplace.m
	 * Note that W is NOT the diagonal matrix
	 */
	SGMatrix<float64_t> m_Rvdd;

	/** V defined in infFITC.m and infFITC_Laplace.m */
	SGMatrix<float64_t> m_V;

private:
	/* init */
	void init();
};
}
#endif /* CSINGLEFITCLAPLACIANBASE_H */

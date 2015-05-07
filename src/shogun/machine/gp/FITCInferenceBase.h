/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * Written (W) 2013 Roman Votyakov
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

#ifndef CFITCINFERENCEBASE_H
#define CFITCINFERENCEBASE_H

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/InferenceMethod.h>

namespace shogun
{

/** @brief The Fully Independent Conditional Training inference base class.
 *
 * For more details, see Quiñonero-Candela, Joaquin, and Carl Edward Rasmussen.
 * "A unifying view of sparse approximate Gaussian process regression."
 * The Journal of Machine Learning Research 6 (2005): 1939-1959.
 *
 * The key idea of FITC inference is to use the following kernel matrix \f$\Sigma_{fitc}\f$
 * to approximate a kernel matrix, \f$\Sigma_{N}\f$ derived from a GP prior.
 *\f[
 *\Sigma_{fitc}=\textbf{diag}(\Sigma_{N}-\Phi)+\Phi
 *\f]
 * where
 *\f$\Phi=\Sigma_{NM}\Sigma_{M}^{-1}\Sigma_{MN}\f$
 *\f$\Sigma_{N}\f$ is the kernel matrix on features
 *\f$\Sigma_{M}\f$ is the kernel matrix on inducing points
 *\f$\Sigma_{NM}=\Sigma_{MN}^{T}\f$ is the kernel matrix between features and inducing features
 *
 * Note that the number of inducing points (m) is usually far less than the number of input points (n). (the time complexity is computed based on the assumption m < n)
 * The idea of FITC approximation is to use a lower-ranked matrix plus a diagonal matrix to approximate the full kernel
 * matrix.
 * The time complexity of the main inference process can be reduced from O(n^3) to O(m^2*n).
 *
 * Since we use \f$\Sigma_{fitc}\f$ to approximate \f$\Sigma_{N}\f$,
 * the (approximated) negative log marginal likelihood are computed based on \f$\Sigma_{fitc}\f$.
 *
 */
class CFITCInferenceBase: public CInferenceMethod
{
public:
	/** default constructor */
	CFITCInferenceBase();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 * @param inducing_features features to use
	 */
	CFITCInferenceBase(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model,
			CFeatures* inducing_features);

	virtual ~CFITCInferenceBase();

	/** return what type of inference we are
	 *
	 * @return inference type FITC
	 */
	virtual EInferenceType get_inference_type() const { return INF_FITC; }

	/** returns the name of the inference method
	 *
	 * @return name FITCBase
	 */
	virtual const char* get_name() const { return "FITCBaseInferenceMethod"; }

	/** set inducing features
	 *
	 * @param feat features to set
	 */
	virtual void set_inducing_features(CFeatures* feat)
	{
		SG_REF(feat);
		SG_UNREF(m_inducing_features);
		m_inducing_features=feat;
	}

	/** get inducing features
	 *
	 * @return features
	 */
	virtual CFeatures* get_inducing_features()
	{
		SG_REF(m_inducing_features);
		return m_inducing_features;
	}

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
	virtual SGVector<float64_t> get_alpha();

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
	virtual SGMatrix<float64_t> get_cholesky();

	/** update all matrices */
	virtual void update()=0;

	/** set the noise for inducing points
	 *
	 * @param noise noise for inducing points
	 *
	 * The noise is used to enfore the kernel matrix about the inducing points are positive definite
	 */
	virtual void set_inducing_noise(float64_t noise);

	/** get the noise for inducing points
	 *
	 * @return noise noise for inducing points
	 *
	 */
	virtual float64_t get_inducing_noise();

	/** returns derivative of negative log marginal likelihood wrt inducing features (input)
	 * Note that in order to call this method, kernel must support FITC inference
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inducing_features(const TParameter* param)=0;

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

protected:
	/** convert inducing features and features to the same represention
	 *
	 * Note that these two kinds of features can be different types.
	 * The reasons are listed below.
	 * 1. The type of the gradient wrt inducing features is float64_t, which is used to update inducing features
	 * 2. Reason 1 implies that the type of inducing features can be float64_t while the type of features does not required
	 * as float64_t
	 * 3. Reason 2 implies that the type of features must be a subclass of CDotFeatures, which can represent features as
	 * float64_t
	 */
	virtual void convert_features();

	/** check whether features and inducing features are set
	 */
	virtual void check_features();

	/** check if members of object are valid for inference */
	virtual void check_members() const;

	/** update train kernel matrix */
	virtual void update_train_kernel();

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInferenceMethod class
	 *
	 * @param param parameter of CInferenceMethod class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inference_method(
			const TParameter* param)=0;

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
			const TParameter* param)=0;

	/** returns derivative of negative log marginal likelihood wrt mean
	 * function's parameter
	 *
	 * @param param parameter of given mean function
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_mean(
			const TParameter* param)=0;

	/** returns derivative of negative log marginal likelihood wrt
	 * inducing noise (noise from inducing features) parameter
	 *
	 * @param param parameter of given  FITCInferenceBase class
	 *
	 * In order to enforce symmetrc positive definiteness of the kernel matrix on inducing points,
	 * \f$\Sigma_{M}\f$, the following ridge trick is used since the matrix is learned from data.
	 *
	 * \f[
	 * \Sigma_{M'}=\Sigma_{M}+\lambda*I
	 * \f]
	 * where
	 * \f$\lambda \ge 0\f$ is the inducing noise.
	 *
	 * In practice, we use the corrected matrix, \Sigma_{M'} in the following approximation.
	 *\f[
	 *\Sigma_{fitc}=\textbf{diag}(\Sigma_{N}-\Phi)+\Phi
	 *\f]
	 * where
	 *\f$\Phi=\Sigma_{NM}\Sigma_{M'}^{-1}\Sigma_{MN}\f$
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inducing_noise(
			const TParameter* param)=0;

	/** inducing features for approximation */
	CFeatures* m_inducing_features;

	/** noise of the inducing variables */
	float64_t m_log_ind_noise;

	/** covariance matrix of inducing features */
	SGMatrix<float64_t> m_kuu;

	/** covariance matrix of inducing features and training features */
	SGMatrix<float64_t> m_ktru;

	/** covariance matrix of the the posterior Gaussian distribution */
	SGMatrix<float64_t> m_Sigma;

	/** mean vector of the the posterior Gaussian distribution */
	SGVector<float64_t> m_mu;

	/** diagonal elements of kernel matrix m_ktrtr */
	SGVector<float64_t> m_ktrtr_diag;
private:
	/** init */
	void init();
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CFITCINFERENCEBASE_H */

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
 * Code adapted from
 * http://www.aueb.gr/users/mtitsias/code/varsgp.tar.gz
 */

#ifndef CVARDTCINFERENCEMETHOD_H
#define CVARDTCINFERENCEMETHOD_H


#include <shogun/lib/config.h>
#include <shogun/machine/gp/SingleSparseInference.h>

namespace shogun
{
/** @brief The inference method class based on the Titsias' variational bound.
 * For more details, see Titsias, Michalis K.
 * "Variational learning of inducing variables in sparse Gaussian processes."
 * International Conference on Artificial Intelligence and Statistics. 2009.
 *
 * NOTE: The Gaussian Likelihood Function must be used for this inference
 * method.
 *
 */
class VarDTCInferenceMethod: public SingleSparseInference
{
public:
	/** default constructor */
	VarDTCInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 * @param inducing_features features to use
	 */
	VarDTCInferenceMethod(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model,
			std::shared_ptr<Features> inducing_features);

	~VarDTCInferenceMethod() override;

	/** returns the name of the inference method
	 *
	 * @return name VarDTC
	 */
	const char* get_name() const override { return "VarDTCInferenceMethod"; }

	/** return what type of inference we are
	 *
	 * @return inference type KL_SPARSE_REGRESSION
	 */
	EInferenceType get_inference_type() const override { return INF_KL_SPARSE_REGRESSION; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CVarDTCInferenceMethod object
	 */
	static std::shared_ptr<VarDTCInferenceMethod> obtain_from_generic(const std::shared_ptr<Inference>& inference);

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

	/**
	 * @return whether combination of sparse inference method and given likelihood
	 * function supports regression
	 */
	bool supports_regression() const override
	{
		check_members();
		return m_model->supports_regression();
	}

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
	SGVector<float64_t> get_posterior_mean() override;

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
	SGMatrix<float64_t> get_posterior_covariance() override;

	/** update all matrices */
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

	/** update cholesky Matrix.*/
	void update_chol() override;

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	void update_deriv() override;

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_likelihood_model(
			Parameters::const_reference param) override;

	/** returns derivative of negative log marginal likelihood wrt inducing features (input)
	 * Note that in order to call this method, kernel must support Sparse inference,
	 * which means derivatives wrt inducing features can be computed
	 *
	 * Note that the kernel must support to compute the derivatives wrt inducing features
	 *
	 * @param param parameter of given kernel
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inducing_features(
		Parameters::const_reference param) override;

	/** returns derivative of negative log marginal likelihood wrt inducing noise
	 *
	 * @param param parameter of given inference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inducing_noise(
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

	/** update gradients */
	void compute_gradient() override;
protected:
	/** inv_Lm=inv(Lm) where Lm*Lm'=Kmm */
	SGMatrix<float64_t> m_inv_Lm;
	/** Knm*inv_Lm */
	SGMatrix<float64_t> m_Knm_inv_Lm;
	/** invLa=inv(La) where La*La'=sigma2*eye(m)+inv_Lm*Kmn*Knm*inv_Lm' */
	SGMatrix<float64_t> m_inv_La;
	/** yy=(y-meanfun)'*(y-meanfun) */
	float64_t m_yy;
	/** the term used to compute gradient wrt likelihood and marginal likelihood*/
	float64_t m_f3;
	/** square of sigma from Gaussian likelihood*/
	float64_t m_sigma2;
	/** the trace term to compute marginal likelihood*/
	float64_t m_trk;
	/** a matrix used to compute gradients wrt kernel (Kmm)*/
	SGMatrix<float64_t> m_Tmm;
	/** a matrix used to compute gradients wrt kernel (Knm)*/
	SGMatrix<float64_t> m_Tnm;
private:
	/** init */
	void init();
};
}
#endif /* CVARDTCINFERENCEMETHOD_H */

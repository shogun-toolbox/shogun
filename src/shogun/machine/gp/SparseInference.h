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

#ifndef CSPARSEINFERENCE_H
#define CSPARSEINFERENCE_H

#include <shogun/lib/config.h>


#include <shogun/machine/gp/Inference.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief The Fully Independent Conditional Training inference base class.
 *
 * For more details, see Quiñonero-Candela, Joaquin, and Carl Edward Rasmussen.
 * "A unifying view of sparse approximate Gaussian process regression."
 * The Journal of Machine Learning Research 6 (2005): 1939-1959.
 *
 * The key idea of Sparse inference is to use the following kernel matrix \f$\Sigma_{fitc}\f$
 * to approximate a kernel matrix, \f$\Sigma_{N}\f$ derived from a GP prior.
 *\f[
 *\Sigma_{Sparse}=\textbf{diag}(\Sigma_{N}-\Phi)+\Phi
 *\f]
 * where
 *\f$\Phi=\Sigma_{NM}\Sigma_{M}^{-1}\Sigma_{MN}\f$
 *\f$\Sigma_{N}\f$ is the kernel matrix on features
 *\f$\Sigma_{M}\f$ is the kernel matrix on inducing points
 *\f$\Sigma_{NM}=\Sigma_{MN}^{T}\f$ is the kernel matrix between features and inducing features
 *
 * Note that the number of inducing points (m) is usually far less than the number of input points (n). (the time complexity is computed based on the assumption m < n)
 * The idea of Sparse approximation is to use a lower-ranked matrix plus a diagonal matrix to approximate the full kernel
 * matrix.
 * The time complexity of the main inference process can be reduced from O(n^3) to O(m^2*n).
 *
 * Since we use \f$\Sigma_{Sparse}\f$ to approximate \f$\Sigma_{N}\f$,
 * the (approximated) negative log marginal likelihood are computed based on \f$\Sigma_{Sparse}\f$.
 *
 */
class SparseInference: public Inference
{
public:
	/** default constructor */
	SparseInference();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 * @param inducing_features features to use
	 */
	SparseInference(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model,
			std::shared_ptr<Features> inducing_features);

	~SparseInference() override;

	/** return what type of inference we are
	 *
	 * @return inference type Sparse
	 */
	EInferenceType get_inference_type() const override { return INF_SPARSE; }

	/** returns the name of the inference method
	 *
	 * @return name SparseBase
	 */
	const char* get_name() const override { return "SparseBaseInferenceMethod"; }

	/** set inducing features
	 *
	 * @param feat features to set
	 */
	virtual void set_inducing_features(std::shared_ptr<Features> feat)
	{
		require(feat,"Input inducing features must be not empty");
		m_inducing_features = feat;
	}

	/** get inducing features
	 *
	 * @return features
	 */
	virtual std::shared_ptr<Features> get_inducing_features()
	{
		auto inducing_feat = m_inducing_features->as<DotFeatures>();
		return std::make_shared<DenseFeatures<float64_t>>(inducing_feat);
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

	/** update all matrices */
	void update() override =0;

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

#ifndef SWIG
	/** returns derivative of negative log marginal likelihood wrt inducing features (input)
	 * Note that in order to call this method, kernel must support Sparse inference
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inducing_features(Parameters::const_reference param)=0;
#endif

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
	SGVector<float64_t> get_posterior_mean() override =0;

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
	SGMatrix<float64_t> get_posterior_covariance() override =0;

protected:
	/** convert inducing features and features to the same represention
	 *
	 * Note that these two kinds of features can be different types.
	 * The reasons are listed below.
	 * 1. The type of the gradient wrt inducing features is float64_t, which is used to update inducing features
	 * 2. Reason 1 implies that the type of inducing features can be float64_t while the type of features does not required
	 * as float64_t
	 * 3. Reason 2 implies that the type of features must be a subclass of DotFeatures, which can represent features as
	 * float64_t
	 */
	virtual void convert_features();

	/** check whether features and inducing features are set
	 */
	virtual void check_features();

	/** check if members of object are valid for inference */
	void check_members() const override;

	/** update train kernel matrix */
	void update_train_kernel() override;

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInference class
	 *
	 * @param param parameter of CInference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inference_method(
			Parameters::const_reference param) override =0;

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_likelihood_model(
			Parameters::const_reference param) override =0;

	/** returns derivative of negative log marginal likelihood wrt kernel's
	 * parameter
	 *
	 * @param param parameter of given kernel
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_kernel(
			Parameters::const_reference param) override =0;

	/** returns derivative of negative log marginal likelihood wrt mean
	 * function's parameter
	 *
	 * @param param parameter of given mean function
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_mean(
			Parameters::const_reference param) override =0;

	/** returns derivative of negative log marginal likelihood wrt
	 * inducing noise (noise from inducing features) parameter
	 *
	 * @param param parameter of given  SparseInference class
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
	 * In practice, we use the corrected matrix, \f$\Sigma_{M'}\f$ in the following approximation.
	 *\f[
	 *\Sigma_{Sparse}=\textbf{diag}(\Sigma_{N}-\Phi)+\Phi
	 *\f]
	 * where
	 *\f$\Phi=\Sigma_{NM}\Sigma_{M'}^{-1}\Sigma_{MN}\f$
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inducing_noise(
			Parameters::const_reference param)=0;

	/** inducing features for approximation */
	std::shared_ptr<Features> m_inducing_features;

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
#endif /* CSPARSEINFERENCE_H */

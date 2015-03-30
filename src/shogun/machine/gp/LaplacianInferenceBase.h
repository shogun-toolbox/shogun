/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
#ifndef CLAPLACIANINFERENCEBASE_H_
#define CLAPLACIANINFERENCEBASE_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/InferenceMethod.h>

namespace shogun
{

/** @brief The Laplace approximation inference method base class
 *
 * This inference method approximates the posterior likelihood function by using
 * Laplace's method. Here, we compute a Gaussian approximation to the posterior
 * via a Taylor expansion around the maximum of the posterior likelihood
 * function.
 *
 */
class CLaplacianInferenceBase: public CInferenceMethod
{
public:
	/** default constructor */
	CLaplacianInferenceBase();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CLaplacianInferenceBase(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CLaplacianInferenceBase();

	/** return what type of inference we are
	 *
	 * @return inference type Laplacian
	 */
	virtual EInferenceType get_inference_type() const { return INF_LAPLACIAN; }

	/** returns the name of the inference method
	 *
	 * @return name Laplacian
	 */
	virtual const char* get_name() const { return "LaplacianInferenceBase"; }

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 *
	 * \f[
	 * \mu = K\alpha+meanf
	 * \f]
	 *
	 * where \f$\mu\f$ is the mean,
	 * \f$K\f$ is the prior covariance matrix,
	 * and \f$meanf$\f is the mean prior fomr MeanFunction
	 *
	 */
	virtual SGVector<float64_t> get_alpha();

	/** get Cholesky decomposition matrix
	 *
	 * @return Cholesky decomposition of matrix:
	 * 
	 *
	 * for binary and regression case
	 * \f[
	 * L = Cholesky(W^{\frac{1}{2}}*K*W^{\frac{1}{2}}+I)
	 * \f]
	 *
	 * where \f$K\f$ is the prior covariance matrix, \f$sW\f$ is the vector
	 * returned by get_diagonal_vector(), and \f$I\f$ is the identity matrix.
	 *
	 * for multiclass case
	 * \f[
	 * M = Cholesky(\sum_\text{c}{E_\text{c})
	 * \f]
	 *
	 * where \f$E_\text{c}\f$ is the matrix defined in the algorithm 3.3 of the GPML textbook for class c
	 * Note the E matrix is used to store these E_\text{c} matrices, where E=[E_1, E_2, ..., E_C], 
	 * where C is the number of classes and C should be greater than 1.
	 */
	virtual SGMatrix<float64_t> get_cholesky();

	/** returns covariance matrix \f$\Sigma=(K^{-1}+W)^{-1}\f$ of the Gaussian
	 * distribution \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to
	 * the posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 * @return covariance matrix
	 */
	virtual SGMatrix<float64_t> get_posterior_covariance();

	/** update data all matrices */
	virtual void update();

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

private:
	/** init */
	void init();

protected:

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov()=0;

	/** derivative of log likelihood with respect to function location */
	SGVector<float64_t> m_dlp;

	/** noise matrix */
	SGVector<float64_t> m_W;

	/** mean vector of the approximation to the posterior */
	SGVector<float64_t> m_mu;

	/** covariance matrix of the approximation to the posterior */
	SGMatrix<float64_t> m_Sigma;

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
#endif /* HAVE_EIGEN3 */
#endif /* CLAPLACIANINFERENCEBASE_H_ */

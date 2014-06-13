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
 * Challis, Edward, and David Barber.
 * "Concave Gaussian variational approximations for inference in large-scale Bayesian linear models."
 * International conference on Artificial Intelligence and Statistics. 2011.
 *
 */

#ifndef _KLCHOLESKYINFERENCEMETHOD_H_
#define _KLCHOLESKYINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/machine/gp/KLInferenceMethod.h>
#include <shogun/mathematics/eigen3.h>

namespace shogun
{

/** @brief The KL approximation inference method class.
 *
 * The class is implemented based on the KL method in the Challis's paper
 *
 * Note that "Cholesky" means Cholesky represention of the variational co-variance matrix
 * is explicitly used in inference
 */
class CKLCholeskyInferenceMethod: public CKLInferenceMethod
{
public:
	/** default constructor */
	CKLCholeskyInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CKLCholeskyInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CKLCholeskyInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name KLCholeskyInferenceMethod
	 */
	virtual const char* get_name() const { return "KLCholeskyInferenceMethod"; }

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 */
	virtual SGVector<float64_t> get_alpha();

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix:
	 *
	 * Note that this vector is not avaliable for the KL method 
	 */
	virtual SGVector<float64_t> get_diagonal_vector();
protected:
	/** update alpha vector */
	virtual void update_alpha();

	/** update cholesky matrix */
	virtual void update_chol();

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	virtual void update_deriv();

	/** the helper function to compute
	 * the negative log marginal likelihood
	 *
	 * @return negative log marginal likelihood
	 */
	virtual float64_t get_negative_log_marginal_likelihood_helper();

	/** compute the gradient wrt variational parameters
	 * given the current variational parameters (mu and s2)
	 *
	 * @return gradient of negative log marginal likelihood
	 */
	virtual void get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient);

	/** pre-compute the information for lbfgs optimization.
	 * This function needs to be called before calling
	 * get_negative_log_marginal_likelihood_wrt_parameters()
	 * and/or
	 * get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient)
	 *
	 */
	virtual void lbfgs_precompute();

	/** compute matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt  hyperparameter in cov function
	 * Note that 
	 * get_derivative_wrt_inference_method(const TParameter* param)
	 * and
	 * get_derivative_wrt_kernel(const TParameter* param)
	 * will call this function
	 *
	 * @param the gradient wrt hyperparameter related to cov
	 */
	virtual float64_t get_derivative_related_cov(Eigen::MatrixXd eigen_dK);

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov();

private:
	void init();

	/** Update the lower triangular part of C
	 */
	void update_C();

	/** Conver the lower triangular part of a matrix to a vector
	 */
	void get_lower_triangular_vector(SGMatrix<float64_t> square_matrix, SGVector<float64_t> target);


	/** The linear solver used to compute kernel inverse
	 */
	Eigen::LDLT<Eigen::MatrixXd> m_ldlt;

	/** The Cholesky represention of the variational co-variance matrix
	 *  Note that Sigma=CC', where C is a lower triangular matrix, C is NOT unique
	 */
	SGMatrix<float64_t> m_C;

	/** The K^{-1}C matrix
	 */
	SGMatrix<float64_t> m_InvK_C; 

	/** The K^{-1}CC' matrix
	 */
	SGMatrix<float64_t> m_InvK_Sigma;

	/** The mean vector generated from mean function
	 */
	SGVector<float64_t> m_mean_vec;

	/** The Log-Determinant of Kernel 
	 */
	float64_t m_log_det_Kernel;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _KLCHOLESKYINFERENCEMETHOD_H_ */

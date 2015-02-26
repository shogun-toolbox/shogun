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

#ifndef _KLLOWERTRIANGULARINFERENCEMETHOD_H_
#define _KLLOWERTRIANGULARINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/machine/gp/KLInferenceMethod.h>

namespace shogun
{

/** @brief The KL approximation inference method class.
 *
 * The class is implemented based on the KL method in the Challis's paper,
 * which uses lower triangular represention.
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
 * Note that "lowerTriangular" means lowerTriangular represention of the variational co-variance matrix
 * is explicitly used in inference
 */
class CKLLowerTriangularInferenceMethod: public CKLInferenceMethod
{
public:
	/** default constructor */
	CKLLowerTriangularInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CKLLowerTriangularInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CKLLowerTriangularInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name KLLowerTriangularInferenceMethod
	 */
	virtual const char* get_name() const { return "KLLowerTriangularInferenceMethod"; }

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix:
	 *
	 * Note that this vector is not avaliable for the KL method 
	 */
	virtual SGVector<float64_t> get_diagonal_vector();

protected:
	/** update cholesky matrix */
	virtual void update_chol();

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	virtual void update_deriv();

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

	virtual float64_t get_derivative_related_cov(SGMatrix<float64_t> dK);

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov();

	/** The K^{-1}Sigma matrix */
	SGMatrix<float64_t> m_InvK_Sigma;

	/** The mean vector generated from mean function */
	SGVector<float64_t> m_mean_vec;

	/** The Log-determinant of Kernel */
	float64_t m_log_det_Kernel;

	/**The L*sqrt(D) matrix, where L and D are defined in LDLT factorization on Kernel*sq(m_scale) */
	SGMatrix<float64_t> m_Kernel_LsD; 

	/**The permutation sequence of P, where P are defined in LDLT factorization on Kernel*sq(m_scale) */
	SGVector<index_t> m_Kernel_P; 

	/** compute the inv(corrected_Kernel*sq(m_scale))*A
	 *
	 * @param A input matrix
	 *
	 * @return inv(corrected_Kernel*sq(m_scale))*A:
	 */
	Eigen::MatrixXd solve_inverse(Eigen::MatrixXd A);

	/** correct the kernel matrix and factorizated the corrected Kernel matrix
	 * for update
	 */
	virtual void update_init();

	/** compute posterior Sigma matrix*/
	virtual void update_Sigma()=0;

	/** compute inv(corrected_Kernel)*Sigma matrix */
	virtual void update_InvK_Sigma()=0;

private:
	void init();

};
}
#endif /* HAVE_EIGEN3 */
#endif /* _KLLOWERTRIANGULARINFERENCEMETHOD_H_ */

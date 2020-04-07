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

#include <shogun/machine/gp/KLLowerTriangularInference.h>

namespace shogun
{

/** @brief The KL approximation inference method class.
 *
 * The class is implemented based on the KL method in the Challis's paper,
 * which uses full Cholesky represention.
 * Note that C is not unique according to the definition of C in the paper.
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
 * The adapted Matlab code can be found at
 * https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
 *
 * Note that "Cholesky" means Cholesky represention of the variational co-variance matrix
 * is explicitly used in inference
 */
class KLCholeskyInferenceMethod: public KLLowerTriangularInference
{
public:
	/** default constructor */
	KLCholeskyInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	KLCholeskyInferenceMethod(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model);

	~KLCholeskyInferenceMethod() override;

	/** returns the name of the inference method
	 *
	 * @return name KLCholeskyInferenceMethod
	 */
	const char* get_name() const override { return "KLCholeskyInferenceMethod"; }

	/** return what type of inference we are
	 *
	 * @return inference type KL_CHOLESKY
	 */
	EInferenceType get_inference_type() const override { return INF_KL_CHOLESKY; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted KLCholeskyInferenceMethod object
	 */
	static std::shared_ptr<KLCholeskyInferenceMethod> obtain_from_generic(const std::shared_ptr<Inference>& inference);

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 */
	SGVector<float64_t> get_alpha() override;

protected:
	/** update alpha vector */
	void update_alpha() override;

	/** the helper function to compute
	 * the negative log marginal likelihood
	 *
	 * @return negative log marginal likelihood
	 */
	float64_t get_negative_log_marginal_likelihood_helper() override;

	/** compute the gradient wrt variational parameters
	 * given the current variational parameters (mu and s2)
	 *
	 * @return gradient of negative log marginal likelihood
	 */
	void get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient) override;

	/** pre-compute the information for optimization.
	 * This function needs to be called before calling
	 * get_negative_log_marginal_likelihood_wrt_parameters()
	 * and/or
	 * get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient)
	 *
	 * @return true if precomputed parameters are valid
	 *
	 */
	bool precompute() override;

	/** compute posterior Sigma matrix*/
	void update_Sigma() override;

	/** compute inv(corrected_Kernel)*Sigma matrix */
	void update_InvK_Sigma() override;
private:
	void init();

	/** Update the lower triangular part of C
	 */
	void update_C();

	/** Conver the lower triangular part of a matrix to a vector
	 */
	void get_lower_triangular_vector(SGMatrix<float64_t> square_matrix, SGVector<float64_t> target);

	/** The Cholesky represention of the variational co-variance matrix
	 *  Note that Sigma=CC', where C is a lower triangular matrix, C is NOT unique
	 */
	SGMatrix<float64_t> m_C;

	/** The K^{-1}C matrix */
	SGMatrix<float64_t> m_InvK_C;

};
}
#endif /* _KLCHOLESKYINFERENCEMETHOD_H_ */

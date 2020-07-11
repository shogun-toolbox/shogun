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


#ifndef CSINGLESPARSEINFERENCE_H
#define CSINGLESPARSEINFERENCE_H

#include <shogun/lib/config.h>
#include <shogun/machine/gp/SparseInference.h>
#include <shogun/optimization/FirstOrderMinimizer.h>
#include <shogun/lib/Lock.h>

namespace shogun
{
class SingleSparseInferenceCostFunction;

/** @brief The sparse inference base class
 * for classification and regression for 1-D labels (1D regression and binary classification)
 */
class SingleSparseInference: public SparseInference
{
friend class SingleSparseInferenceCostFunction;

public:
	/** default constructor */
	SingleSparseInference();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 * @param inducing_features features to use
	 */
	SingleSparseInference(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model,
			std::shared_ptr<Features> inducing_features);

	~SingleSparseInference() override;

	/** returns the name of the inference method
	 *
	 * @return name SingleSparseInference
	 */
	const char* get_name() const override { return "SingleSparseInference"; }

	/** set kernel
	 *
	 * @param kern kernel to set
	 */
	void set_kernel(std::shared_ptr<Kernel> kern) override;

	/** opitmize inducing features
	 *
	 */
	virtual void optimize_inducing_features();

	/** set the lower bound of inducing features
	 *
	 * @param bound lower bound constrains of inducing features
	 *
	 * Note that if the length of the bound is 1,
	 * it means the bound constraint applies to each dimension of all inducing features
	 *
	 * Note that if the length of the bound is greater than 1,
	 * it means each dimension of the bound constraint applies to the corresponding dimension of inducing features
	 */
	virtual void set_lower_bound_of_inducing_features(SGVector<float64_t> bound);

	/** set the upper bound of inducing features
	 *
	 * @param bound upper bound constrains of inducing features
	 *
	 * Note that if the length of the bound is 1,
	 * it means the bound constraint applies to each dimension of all inducing features
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
	 * @param minimizer minimizer used in optimization
	 */
	virtual void enable_optimizing_inducing_features(bool is_optmization, std::shared_ptr<FirstOrderMinimizer> minimizer=NULL);

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
		SGMatrix<float64_t> dKuui, SGMatrix<float64_t> dKui)=0;

	/** returns derivative of negative log marginal likelihood wrt inducing noise
	 *
	 * @param param parameter of given inference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inducing_noise(
		Parameters::const_reference param) override =0;


	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInference class
	 *
	 * @param param parameter of CInference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inference_method(
			Parameters::const_reference param) override;


	/** returns derivative of negative log marginal likelihood wrt kernel's
	 * parameter
	 *
	 * @param param parameter of given kernel
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_kernel(
			Parameters::const_reference param) override;

	/** check the bound constraint is vailid or not
	 *
	 * @param bound bound constrains of inducing features
	 * @param name the name of the bound
	 */
	virtual void check_bound(SGVector<float64_t> bound, const char* name);

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

	/** check whether the provided kernel can
	 * compute the gradient wrt inducing features
	 *
	 * Note that currently we check the name of the provided kernel
	 * to determine whether the kernel can compute the derivatives wrt inducing_features
	 *
	 * The name of a supported Kernel must end with "SparseKernel"
	 */
	virtual void check_fully_sparse();

	/** returns derivative of negative log marginal likelihood wrt inducing features (input)
	 * Note that in order to call this method, kernel must support Sparse inference,
	 * which means derivatives wrt inducing features can be computed
	 *
	 * Note that the kernel must support to compute the derivatives wrt inducing features
	 *
	 * @param param parameter of given kernel
	 * @return derivative of negative log marginal likelihood
	 */
	SGVector<float64_t> get_derivative_wrt_inducing_features(Parameters::const_reference param) override =0;

	/** whether the kernel supports to get the gradient wrt inducing points or not*/
	bool m_fully_sparse;

	/** a lock used to parallelly compute derivatives wrt hyperparameters */
	Lock m_lock;

	/** minimizer used in finding optimal inducing features*/
	std::shared_ptr<FirstOrderMinimizer> m_inducing_minimizer;
private:
	/* init */
	void init();
};
}
#endif /* CSINGLESPARSEINFERENCE_H */

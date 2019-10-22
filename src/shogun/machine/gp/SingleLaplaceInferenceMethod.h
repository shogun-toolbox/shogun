/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wu Lin
 */

#ifndef CSINGLELAPLACEINFERENCEMETHOD_H_
#define CSINGLELAPLACEINFERENCEMETHOD_H_

#include <shogun/lib/config.h>
#include <shogun/machine/gp/LaplaceInference.h>
#include <shogun/optimization/Minimizer.h>

namespace shogun
{
/** @brief The SingleLaplace approximation inference method class
 * for regression and binary Classification.
 *
 * This inference method approximates the posterior likelihood function by using
 * Laplace's method. Here, we compute a Gaussian approximation to the posterior
 * via a Taylor expansion around the maximum of the posterior likelihood
 * function.
 *
 * For more details, see "Bayesian Classification with Gaussian Processes" by
 * Christopher K.I Williams and David Barber, published 1998 in the IEEE
 * Transactions on Pattern Analysis and Machine Intelligence, Volume 20, Number
 * 12, Pages 1342-1351.
 *
 * This specific implementation was adapted from the infLaplace.m file in the
 * GPML toolbox.
 */
class SingleLaplaceInferenceMethod: public LaplaceInference
{
friend class SingleLaplaceNewtonOptimizer; 
friend class SingleLaplaceInferenceMethodCostFunction;
public:
	/** default constructor */
	SingleLaplaceInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	SingleLaplaceInferenceMethod(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
			std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model);

	virtual ~SingleLaplaceInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name SingleLaplace
	 */
	virtual const char* get_name() const { return "SingleLaplaceInferenceMethod"; }

	/** return what type of inference we are
	 *
	 * @return inference type Laplace_Single
	 */
	virtual EInferenceType get_inference_type() const { return INF_LAPLACE_SINGLE; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted SingleLaplaceInferenceMethod object
	 */
	static std::shared_ptr<SingleLaplaceInferenceMethod> obtain_from_generic(std::shared_ptr<Inference> inference);

	/** get negative log marginal likelihood
	 *
	 * @return the negative log of the marginal likelihood function:
	 *
	 * \f[
	 * -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and
	 * \f$\theta\f$ represent hyperparameters.
	 */
	virtual float64_t get_negative_log_marginal_likelihood();

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports regression
	 */
	virtual bool supports_regression() const
	{
		check_members();
		return m_model->supports_regression();
	}

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports binary classification
	 */
	virtual bool supports_binary() const
	{
		check_members();
		return m_model->supports_binary();
	}

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
	virtual SGVector<float64_t> get_diagonal_vector();

	/** update all matrices except gradients*/
	virtual void update();

	/** returns mean vector \f$\mu\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 * Mean vector \f$\mu\f$ is evaluated using Newton's method.
	 *
	 * @return mean vector
	 */
	virtual SGVector<float64_t> get_posterior_mean();

	  
	/** Set a minimizer
	 *
	 * @param minimizer minimizer used in inference method
	 */
	virtual void register_minimizer(std::shared_ptr<Minimizer> minimizer);
protected:

	/** initialize the update  */
	virtual void update_init();

	/** update alpha matrix */
	virtual void update_alpha();

	/** update cholesky matrix */
	virtual void update_chol();

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov();

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	virtual void update_deriv();

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInference class
	 *
	 * @param param parameter of CInference class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inference_method(
			Parameters::const_reference param);

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_likelihood_model(
			Parameters::const_reference param);

	/** returns derivative of negative log marginal likelihood wrt kernel's
	 * parameter
	 *
	 * @param param parameter of given kernel
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_kernel(
			Parameters::const_reference param);

	/** returns derivative of negative log marginal likelihood wrt mean
	 * function's parameter
	 *
	 * @param param parameter of given mean function
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_mean(
			Parameters::const_reference param);

	/** compute the function value given the current alpha
	 *
	 * @return the function value
	 */
	float64_t get_psi_wrt_alpha();

	/** compute the gradient given the current alpha
	 *
	 * @param gradient derivative of the function wrt alpha
	 */
	void get_gradient_wrt_alpha(SGVector<float64_t> gradient);

private:
	void init();

protected:
	/** a parameter used to compute function value and gradient for LBFGS update*/
	SGVector<float64_t> m_mean_f;

	/** square root of W */
	SGVector<float64_t> m_sW;

	/** second derivative of log likelihood with respect to function location */
	SGVector<float64_t> m_d2lp;

	/** third derivative of log likelihood with respect to function location */
	SGVector<float64_t> m_d3lp;

	/** derivative of negative log (approximated) marginal likelihood wrt fhat */ 
	SGVector<float64_t> m_dfhat;

	/** z */
	SGMatrix<float64_t> m_Z;

	/** g */
	SGVector<float64_t> m_g;

	/** posterior log likelihood without constant terms */
	float64_t m_Psi;
};


/** @brief The build-in minimizer for SingleLaplaceInference */
class SingleLaplaceNewtonOptimizer: public Minimizer
{
public:
	SingleLaplaceNewtonOptimizer() :Minimizer() {  init(); }

	virtual const char* get_name() const { return "SingleLaplaceNewtonOptimizer"; }

	virtual ~SingleLaplaceNewtonOptimizer() {  }

	/** Set the inference method
	 * @param obj the inference method
	 */
	void set_target(std::shared_ptr<SingleLaplaceInferenceMethod >obj);

	/** Unset the inference method
	 * @param is_unref do we SG_UNREF the method
	 */
	void unset_target(bool is_unref);

	/** Do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	virtual float64_t minimize();

	/** set maximum for Brent's minimization method
	 *
	 * @param max maximum for Brent's minimization method
	 */
	virtual void set_minimization_max(float64_t max) { m_opt_max=max; }

	/** set tolerance for Brent's minimization method
	 *
	 * @param tol tolerance for Brent's minimization method
	 */
	virtual void set_minimization_tolerance(float64_t tol) { m_opt_tolerance=tol; }

	/** set max Newton iterations
	 *
	 * @param iter max Newton iterations
	 */
	virtual void set_newton_iterations(int32_t iter) { m_iter=iter; }

	/** set tolerance for newton iterations
	 *
	 * @param tol tolerance for newton iterations to set
	 */
	virtual void set_newton_tolerance(float64_t tol) { m_tolerance=tol; }

private:
	void init();

	/** the inference method */
	std::shared_ptr<SingleLaplaceInferenceMethod >m_obj;

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
#endif /* CSINGLELAPLACEINFERENCEMETHOD_H_ */

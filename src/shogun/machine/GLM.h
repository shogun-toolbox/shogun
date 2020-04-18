/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Tej Sukhatme
 */


#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/IterativeMachine.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/distributions/Distribution.h>

namespace shogun
{
    enum GLM_DISTRIBUTION
	{
        //TODO GAUSSIAN,
        //TODO BINOMIAL,
        //TODO GAMMA,
        //TODO SOFTPLUS,
        //TODO PROBIT,
        POISSON
	};

class DotFeatures;
class Features;
class RegressionLabels;

/** @brief Class for estimating regularized generalized linear models (GLM).
 *   The regularized GLM minimizes the penalized negative log likelihood:
 *      .. math::
 *
 *		\\min_{\\beta_0, \\beta} \\frac{1}{N}
 *		\\sum_{i = 1}^N \\mathcal{L} (y_i, \\beta_0 + \\beta^T x_i)
 *		+ \\lambda [ \\frac{1}{2}(1 - \\alpha) \\mathcal{P}_2 +
 *						\\alpha \\mathcal{P}_1 ]
 *
 *	where :math:`\\mathcal{P}_2` and :math:`\\mathcal{P}_1` are the generalized
 *	L2 (Tikhonov) and generalized L1 (Group Lasso) penalties, given by:
 *
 *	.. math::
 *
 *		\\mathcal{P}_2 = \\|\\Gamma \\beta \\|_2^2 \\
 *		\\mathcal{P}_1 = \\sum_g \\|\\beta_{j,g}\\|_2
 *
 *	where :math:`\\Gamma` is the Tikhonov matrix: a square factorization
 *	of the inverse covariance matrix and :math:`\\beta_{j,g}` is the
 *	:math:`j` th coefficient of group :math:`g`.
 *
 *	The generalized L2 penalty defaults to the ridge penalty when
 *	:math:`\\Gamma` is identity.
 *
 *	The generalized L1 penalty defaults to the lasso penalty when each
 *	:math:`\\beta` belongs to its own group.
 *
 * */
class GLM : public IterativeMachine<LinearMachine>, public RandomMixin<Distribution>
{
 	public:
 		/** default constructor */
 		GLM();

 		/** destructor */
 		virtual ~GLM();

		/** apply linear machine to data
		 * for regression problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL) override;

 		/** Returns the name of the SGSerializable instance.  It MUST BE
 		 *  the CLASS NAME without the prefixed `C'.
 		 *
 		 * @return name of the SGSerializable
 		 */
 		virtual const char* get_name() const { return "GeneralizedLinearMachine"; }

	protected:

		virtual void init_model(std::shared_ptr<Features> data);
		
		virtual void iteration();

	private:

		void init();
		
		/**Conditional intensity function.*/
		virtual SGVector<float64_t> conditional_intensity(SGMatrix<float64_t> X, SGVector<float64_t> w, float64_t bias);

		/** compute gradient of weights */
		virtual SGVector<float64_t> compute_grad_L2_loss_w(SGMatrix<float64_t> X, SGVector<float64_t> y, SGVector<float64_t> w, float64_t bias);

		/** compute gradient of bias */
		virtual float64_t compute_grad_L2_loss_bias(SGMatrix<float64_t> X, SGVector<float64_t> y, SGVector<float64_t> w, float64_t bias);

		/** compute z */
		virtual SGVector<float64_t> compute_z(SGMatrix<float64_t> X, SGVector<float64_t> w, float64_t bias);

		/** conditional non-linear function */
		virtual SGVector<float64_t> non_linearity(SGVector<float64_t> z);

		/** compute gradient of non-linearity */
		virtual SGVector<float64_t> gradient_non_linearity(SGVector<float64_t> z);

		virtual SGVector<float64_t> apply_proximal_operator(SGVector<float64_t> w, float64_t threshold);

	protected:

		/** Distribution type */
		GLM_DISTRIBUTION distribution;

		/** a threshold parameter that linearizes the exp() function above eta. */
		float64_t m_eta = 2.0;

		/** maximum number of iterations for the solver. */
		int m_max_iter = 1000;
		
		/** learning rate for gradient descent. */
		float64_t m_learning_rate = 2e-1;

		/** regularization parameter :math:`\\lambda` of penalty term. */
		float64_t m_lambda = 0.1;

		/** the (n_features, n_features) Tikhonov matrix.
		 * default: NULL, in which case Tau is identity and the L2 penalty is ridge-like */
		SGMatrix<float64_t> m_tau;

		/** the weighting between L1 penalty and L2 penalty term of the loss function. */
		float64_t m_alpha = 0.5;

		/** convergence threshold or stopping criteria. Optimization loop will stop when relative change in parameter norm is below the threshold. */
		float64_t m_tolerance = 1e-6;
};
}
// #endif

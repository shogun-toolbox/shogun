/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Tej Sukhatme
 */


#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>


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
class GeneralizedLinearMachine : public LinearMachine
{
 	public:
 		/** default constructor */
 		GeneralizedLinearMachine();

 		/** destructor */
 		virtual ~GeneralizedLinearMachine();

		/** apply linear machine to data
		 * for regression problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL);

 		/** Returns the name of the SGSerializable instance.  It MUST BE
 		 *  the CLASS NAME without the prefixed `C'.
 		 *
 		 * @return name of the SGSerializable
 		 */
 		virtual const char* get_name() const { return "GeneralizedLinearMachine"; }

	protected:

		/**
		* A specialization of the train_machine method
		* @param feats training data
		*/
		virtual bool train_machine(std::shared_ptr<const DenseFeatures<float64_t>> feats);

		/** apply get outputs
		 *
		 * @param data features to compute outputs
		 * @return outputs
		 */
		virtual SGVector<float64_t> apply_get_outputs(std::shared_ptr<Features> data);

		/**Conditional intensity function.*/
		virtual SGVector<float64_t> conditional_intensity(SGMatrix<float64_t> X, SGVector<float64_t> w, float64_t bias);

		/** predict for one vector */
		virtual SGVector<float64_t> predict(SGMatrix<float64_t> X);

		/** fit model */
		virtual bool fit(SGMatrix<float64_t> X, SGVector<float64_t> y);

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

	private:

		void init();

	protected:

		/** Distribution type */
		GLM_DISTRIBUTION distribution;

		/** specifies if a constant (a.k.a. bias or intercept) should be added to the decision function. */
		bool m_fit_intercept = true;

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
		SGMatrix<float64_t> m_tau = NULL;

		/** the weighting between L1 penalty and L2 penalty term of the loss function. */
		float64_t m_alpha = 0.5;

		/** convergence threshold or stopping criteria. Optimization loop will stop when relative change in parameter norm is below the threshold. */
		float64_t m_tolerance = 1e-6;
};
}
// #endif

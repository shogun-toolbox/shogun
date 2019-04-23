/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Evan Shelhamer,
 *          Fernando Iglesias, Youssef Emad El-Din, Heiko Strathmann
 */

#ifndef _LINEARRIDGEREGRESSION_H__
#define _LINEARRIDGEREGRESSION_H__

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/FeatureDispatchCRTP.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/regression/Regression.h>

namespace shogun
{
	/** @brief Class LinearRidgeRegression implements Ridge Regression - a
	 * regularized least square
	 * method for classification and regression.
	 *
	 * Internally, it is solved via minimizing the following system
	 *
	 * \f[
	 * \frac{1}{2}\left(\sum_{i=1}^N(y_i-({\bf w}\cdot {\bf x}_i))^2 +
	 * \tau||{\bf w}||^2\right)
	 * \f]
	 *
	 * Define \f$X=\left[{\bf x}_{1},\dots{\bf
	 * x}_{N}\right]\in\mathbb{R}^{D\times N}\f$, and
	 * \f$y=[y_{1},\dots,y_{N}]^{\top}\in\mathbb{R}^{N}\f$. Then the
	 * solution boils down to solving a linear system
	 *
	 * \f[
	 * {\bf w}=\left(\tau I_{D}+XX^{\top}\right)^{-1}X^{\top}y
	 * \f]
	 *
	 * If the dimension of the data \f$D\f$ is larger than the number of data
	 * \f$N\f$, the weights are computed via the gram matrix
	 *
	 * \f[
	 * X=\left[{\bf x}_{1},\dots{\bf x}_{N}\right]\in\mathbb{R}^{D\times N}
	 * \f]
	 *
	 * 	 *
	 * In practice, an additional bias term
	 * \f$b=\frac{1}{N}\sum_{i=1}^{N}y_{i}-{\bf w}\cdot\bar{\mathbf{x}}\f$
	 * for
	 * \f$\bar{\mathbf{x}}=\frac{1}{N}\sum_{i=1}^{N}{\bf x}_{i}\f$
	 * can also be included, which effectively centers the \f$X\f$ before
	 * computing the solution.
	 */
	class LinearRidgeRegression : public DenseRealDispatch<LinearRidgeRegression, LinearMachine>
	{
		friend class DenseRealDispatch<LinearRidgeRegression, LinearMachine>;

	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor */
		LinearRidgeRegression();

		/** constructor
		 *
		 * @param tau regularization constant tau
		 * @param data training data
		 * @param lab labels
		 */
		LinearRidgeRegression(float64_t tau, std::shared_ptr<DenseFeatures<float64_t>> data, std::shared_ptr<Labels> lab);
		virtual ~LinearRidgeRegression() {}

		/** set regularization constant
		 *
		 * @param tau new tau
		 */
		inline void set_tau(float64_t tau) { m_tau = tau; };

		/** load regression from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save regression to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** get classifier type
		 *
		 * @return classifier type LinearRidgeRegression
		 */
		virtual EMachineType get_classifier_type()
		{
			return CT_LINEARRIDGEREGRESSION;
		}

		/** @return object name */
		virtual const char* get_name() const { return "LinearRidgeRegression"; }

	protected:
		template <typename T>
		bool train_machine_templated(std::shared_ptr<const DenseFeatures<T>> feats);

	private:
		void init();

	protected:
		/** regularization parameter tau */
		float64_t m_tau;

		/** Whether or not to compute an offset term */
		bool m_use_bias;
};
}
#endif // _LINEARRIDGEREGRESSION_H__

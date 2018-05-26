/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk
 */

#ifndef FIRSTORDERSAGCOSTFUNCTIONINTERFACE_H
#define FIRSTORDERSAGCOSTFUNCTIONINTERFACE_H

#include <stan/math.hpp>
#include <functional>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/optimization/FirstOrderSAGCostFunction.h>
using StanVector = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
namespace shogun
{
	/** @brief The first order stochastic cost function base class for
	 * implementing the SAG Cost function
	 *
	 * The class gives the implementation used in first order stochastic
	 * minimizers
	 *
	 * The cost function must be Written as a finite sample-specific sum of
	 * cost.
	 * For example, least squares cost function,
	 * \f[
	 * f(w)=\frac{ \sum_i{ (y_i-w^T x_i)^2 } }{2}
	 * \f]
	 * where \f$(y_i,x_i)\f$ is the i-th sample,
	 * \f$y_i\f$ is the label and \f$x_i\f$ is the features
	 */
	class FirstOrderSAGCostFunctionInterface : public FirstOrderSAGCostFunction
	{
	public:
		FirstOrderSAGCostFunctionInterface(
		    SGMatrix<float64_t> X, SGMatrix<float64_t> y,
		    StanVector* trainable_parameters,
		    Eigen::Matrix<std::function<stan::math::var(int32_t)>,
		                  Eigen::Dynamic, 1>* cost_for_ith_point,
		    std::function<stan::math::var(StanVector*)>* total_cost);

		FirstOrderSAGCostFunctionInterface(){};

		/** Setter for the training data X */
		virtual void
		set_training_data(SGMatrix<float64_t> X_new, SGMatrix<float64_t> y_new);

		/** Setter for the trainable parameters of the cost function */
		virtual void set_trainable_parameters(StanVector* new_params);

		/** Setter for the cost function definition using stan */
		virtual void set_ith_cost_function(
		    Eigen::Matrix<std::function<stan::math::var(int32_t)>,
		                  Eigen::Dynamic, 1>* new_cost_f);

		/** Setter for the overall cost function */
		virtual void set_cost_function(
		    std::function<stan::math::var(StanVector*)>* total_cost);

		virtual ~FirstOrderSAGCostFunctionInterface();

		/** Initialize to generate a sample sequence
		 *
		 */
		virtual void begin_sample();

		/** Get next sample
		 *
		 * @return false if reach the end of the sample sequence
		 * */
		virtual bool next_sample();

		/** Get the sample gradient value wrt target variables
		 *
		 * WARNING
		 * This method does return
		 * \f$ \frac{\partial f_i(w) }{\partial w} \f$,
		 * instead of
		 * \f$\sum_i{ \frac{\partial f_i(w) }{\partial w} }\f$
		 *
		 * For least squares cost function, that is the value of
		 * \f$\frac{\partial f_i(w) }{\partial w}\f$ given \f$w\f$ is known
		 * where the index \f$i\f$ is obtained by next_sample()
		 *
		 * @return sample gradient of variables
		 */
		virtual SGVector<float64_t> get_gradient();

		/** Get the cost given current target variables
		 *
		 * For least squares, that is the value of \f$f(w)\f$.
		 *
		 * @return cost
		 */
		virtual float64_t get_cost();

		/** Get the sample size
		 *
		 * @return the sample size
		 */
		virtual index_t get_sample_size();

		/** Get the average gradient value wrt target variables
		 *
		 * Note that the average gradient is the mean of sample gradient from
		 * get_gradient()
		 * if samples are generated (uniformly) at random.
		 *
		 * WARNING
		 * This method returns
		 * \f$ \frac{\sum_i^n{ \frac{\partial f_i(w) }{\partial w} }}{n}\f$
		 *
		 * For least squares, that is the value of
		 * \f$ \frac{\frac{\partial f(w) }{\partial w}}{n} \f$ given \f$w\f$ is
		 * known
		 * where \f$f(w)=\frac{ \sum_i^n{ (y_i-w^t x_i)^2 } }{2}\f$
		 *
		 * @return average gradient of target variables
		 */
		virtual SGVector<float64_t> get_average_gradient();

	protected:
		/** X is the training data in column major matrix format */
		SGMatrix<float64_t> m_X;

		/** y is the ground truth, or the correct prediction */
		SGMatrix<float64_t> m_y;

		/** trainable_parameters are the variables that are optimized for */
		StanVector* m_trainable_parameters;

		/** cost_for_ith_point is the cost contributed by each point in the
		 * training data */
		Eigen::Matrix<std::function<stan::math::var(int32_t)>, Eigen::Dynamic,
		              1>* m_cost_for_ith_point;

		/** total_cost is the total cost to be minimized, that in this case is a
		 * form of sum of cost_for_ith_point*/
		std::function<stan::math::var(StanVector*)>* m_total_cost;

		/** index_of_sample is the index of the column in X for the current
		 * sample */
		index_t m_index_of_sample;
	};
}

#endif /* FIRSTORDERSAGCOSTFUNCTIONINTERFACE_H  */

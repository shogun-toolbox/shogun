/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk
 */

#ifndef StanFirstOrderSAGCostFunction_H
#define StanFirstOrderSAGCostFunction_H

#include <stan/math.hpp>
#include <functional>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/range.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/optimization/FirstOrderSAGCostFunction.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>

using StanVector = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
template <class T>
using FunctionReturnsStan = std::function<stan::math::var(T)>;
template <class R>
using FunctionReturnsStanVectorArg = std::function<stan::math::var(const StanVector&, R)>;
template <class S>
using StanFunctionsVector =
    Eigen::Matrix<FunctionReturnsStanVectorArg<S>, Eigen::Dynamic, 1>;
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
	class StanFirstOrderSAGCostFunction : public FirstOrderStochasticCostFunction
	{
	public:
		StanFirstOrderSAGCostFunction(
		    SGMatrix<float64_t> X, SGMatrix<float64_t> y,
		    StanVector& trainable_parameters,
		    StanFunctionsVector<float64_t> cost_for_ith_point,
		    FunctionReturnsStan<const StanVector& > total_cost)
          : m_trainable_parameters(trainable_parameters)
        {
          REQUIRE(X.size() > 0, "Empty X provided");
          REQUIRE(y.size() > 0, "Empty y provided");
          auto num_of_variables = trainable_parameters.rows();
          REQUIRE(
              num_of_variables > 0, "Provided %d variables in the parameters, more "
                                    "than 0 parameters required",
              num_of_variables);

          m_X = X;
          m_y = y;
          m_cost_for_ith_point = cost_for_ith_point;
          m_total_cost = total_cost;
          m_ref_trainable_parameters = SGVector<float64_t>(num_of_variables);
          for (auto i : range(num_of_variables))
          {
            m_ref_trainable_parameters[i] = m_trainable_parameters(i, 0).val();
          }
        }

		/** Setter for the training data X */
		virtual void
		set_training_data(SGMatrix<float64_t> X_new, SGMatrix<float64_t> y_new);

		virtual ~StanFirstOrderSAGCostFunction();

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

		virtual SGVector<float64_t> obtain_variable_reference();

		/** Updates m_trainable_parameters values to m_ref_trainable_parameters
		 */
		void update_stan_vectors_to_reference_values();

	protected:
		/** X is the training data in column major matrix format */
		SGMatrix<float64_t> m_X;

		/** y is the ground truth, or the correct prediction */
		SGMatrix<float64_t> m_y;

		/** trainable_parameters are the variables that are optimized for */
		StanVector& m_trainable_parameters;

		/** cost_for_ith_point is the cost contributed by each point in the
		 * training data */

		StanFunctionsVector<float64_t> m_cost_for_ith_point;

		/** total_cost is the total cost to be minimized, that in this case is a
		 * form of sum of cost_for_ith_point*/
		// std::function<stan::math::var(StanVector*)>* m_total_cost;
		FunctionReturnsStan<const StanVector&> m_total_cost;

		/** Reference values for trainable_parameters so that minimizers can
		 * perform inplace updates */
		SGVector<float64_t> m_ref_trainable_parameters;

		/** index_of_sample is the index of the column in X for the current
		 * sample */
		index_t m_index_of_sample;

  private:
    virtual SGVector<float64_t> get_gradient(index_t index);
	};
}

#endif /* StanFirstOrderSAGCostFunction_H  */

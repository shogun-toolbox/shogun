/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2018 Elfarouk
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

#ifndef FIRSTORDERSAGCOSTFUNCTIONINTERFACE_H
#define FIRSTORDERSAGCOSTFUNCTIONINTERFACE_H

#include <stan/math.hpp>
#include <Eigen/Dense>
#include <shogun/lib/config.h>
#include <shogun/optimization/FirstOrderSAGCostFunction.h>
#include <functional>
#include <vector>


namespace shogun{
  /** @brief The first order stochastic cost function base class for implementing
   *  The SAG Cost function
   *
   * The class gives the implementation used in first order stochastic minimizers
   *
   * The cost function must be Written as a finite sample-specific sum of cost.
   * For example, least squares cost function,
   * \f[
   * f(w)=\frac{ \sum_i{ (y_i-w^T x_i)^2 } }{2}
   * \f]
   * where \f$(y_i,x_i)\f$ is the i-th sample,
   * \f$y_i\f$ is the label and \f$x_i\f$ is the features
   */
  class FirstOrderSAGCostFunctionInterface : public FirstOrderSAGCostFunction{
  public:
    /** Consturctor that initializes X, y, trainable_parameters, and the cost_function */
  	FirstOrderSAGCostFunctionInterface(
  		Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic>* X,
  		Eigen::Matrix<float64_t, 1, Eigen::Dynamic>* y,
  		Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>* trainable_parameters,
  		std::vector< std::function < stan::math::var(int32_t) > >* cost_for_ith_point,
      std::function < stan::math::var(std::vector<stan::math::var>*) >* total_cost
  	);

  	/** Default constructor, use setter helpers to set X, y, trainable_parameters, and
  	*		cost_function
  	*/
  	FirstOrderSAGCostFunctionInterface() {};

  	/** Setter for the training data X */
  	virtual void set_training_data(
  		Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic> * X_new,
  		Eigen::Matrix<float64_t, 1, Eigen::Dynamic>* y_new
  	);

  	/** Setter for the trainable parameters of the cost function */
  	virtual void set_trainable_parameters(
  		Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>* new_params
  	);

  	/** Setter for the cost function definition using stan */
  	virtual void set_ith_cost_function(std::vector< std::function < stan::math::var(int32_t) > >* new_cost_f);

    /** Setter for the overall cost function */
    virtual void set_cost_function(std::function < stan::math::var(std::vector<stan::math::var>*) >* total_cost);


  	~FirstOrderSAGCostFunctionInterface();

  	/** Initialize to generate a sample sequence
  	 *
  	 */
  	virtual void begin_sample();

  	/** Get next sample
  	 *
  	 * @return false if reach the end of the sample sequence
  	 * */
  	virtual bool next_sample();

  	/** Get the SAMPLE gradient value wrt target variables
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
    virtual int32_t get_sample_size();

    /** Get the AVERAGE gradient value wrt target variables
     *
     * Note that the average gradient is the mean of sample gradient from get_gradient()
     * if samples are generated (uniformly) at random.
     *
     * WARNING
     * This method returns
     * \f$ \frac{\sum_i^n{ \frac{\partial f_i(w) }{\partial w} }}{n}\f$
     *
     * For least squares, that is the value of
     * \f$ \frac{\frac{\partial f(w) }{\partial w}}{n} \f$ given \f$w\f$ is known
     * where \f$f(w)=\frac{ \sum_i^n{ (y_i-w^t x_i)^2 } }{2}\f$
     *
     * @return average gradient of target variables
     */
    virtual SGVector<float64_t> get_average_gradient();


  protected:
  	/** X is the training data in column major matrix format */
  	Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic>* m_X;

  	/** y is the ground truth, or the correct prediction */
  	Eigen::Matrix<float64_t, 1, Eigen::Dynamic>* m_y;

  	/** trainable_parameters are the variables that are optimized for */
  	Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>* m_trainable_parameters;

    /** cost_for_ith_point is the cost contributed by each point in the training data */
    std::vector< std::function < stan::math::var(int32_t) > >* m_cost_for_ith_point;

    /** total_cost is the total cost to be minimized, that in this case is a form of sum of cost_for_ith_point*/
    std::function < stan::math::var(std::vector<stan::math::var>*) >* m_total_cost;

  	/** index_of_sample is the index of the column in X for the current sample */
  	index_t m_index_of_sample;
  };
}

#endif /* FIRSTORDERSAGCOSTFUNCTIONINTERFACE_H  */

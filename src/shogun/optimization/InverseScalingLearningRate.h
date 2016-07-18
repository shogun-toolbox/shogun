/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
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

#ifndef INVERSESCALINGLEARNINGRATE_H
#define INVERSESCALINGLEARNINGRATE_H
#include <shogun/optimization/LearningRate.h>

namespace shogun
{
/** @brief The implements the inverse scaling learning rate.
 *
 * The learning rate is computed in the following way:
 * \f[  
 * \frac{\eta_0}{{(a+b \times iter)}^k} 
 * \f]
 * where \f$\eta_0\f$ is the initial learning rate,
 * \f$a\f$ is the intercept term,
 * \f$b\f$ is the slope term,
 * \f$iter\f$ is the number of times to call get_learning_rate(),
 * and \f$k\f$ is the exponent term.
 *
 */
class InverseScalingLearningRate: public LearningRate
{
public:
	/*  Constructor */
	InverseScalingLearningRate():LearningRate() { init(); }

	/*  Destructor */
	virtual ~InverseScalingLearningRate() {}

	/** returns the name of the class
	 *           
	 * @return name InverseScalingLearningRate
	 */
	virtual const char* get_name() const { return "InverseScalingLearningRate"; }

	/** Get the learning rate for descent direction
	 * @param iter_counter the number of iterations
	 *
	 * @return the learning rate (A.K.A step size/length)
	 */
	virtual float64_t get_learning_rate(int32_t iter_counter);

	/** Set the initial learning rate
	 *
	 * @param initial_learning_rate initial_learning_rate must be positive
	 */
	virtual void set_initial_learning_rate(float64_t initial_learning_rate);

	/** Set the exponent term
	 *
	 * @param exponent exponent term should be positive
	 */
	virtual void set_exponent(float64_t exponent);

	/** Set the slope term
	 *
	 * @param slope slope term should be positive
	 */
	virtual void set_slope(float64_t slope);

	/** Set the intercept term
	 *
	 * @param intercept intercept term should be positive
	 */
	virtual void set_intercept(float64_t intercept);

protected:
	/**  exponent */
	float64_t m_exponent;
	/**  slope */
	float64_t m_slope;
	/**  intercept */
	float64_t m_intercept;
	/**  init_learning_rate */
	float64_t m_initial_learning_rate;
private:
	/**  Init */
	void init();
};

}

#endif

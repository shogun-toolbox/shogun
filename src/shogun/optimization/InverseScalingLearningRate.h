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
#include <shogun/mathematics/Math.h>

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
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class InverseScalingLearningRate: public LearningRate
{
public:
	/*  Constructor */
	InverseScalingLearningRate():LearningRate() { init(); }

	/*  Destructor */
	virtual ~InverseScalingLearningRate() {}

	/** Get the learning rate for descent direction
	 * Note that every time this method is called, the call counter will increase by 1
	 * Note that the learning rate usually is positive
	 *
	 * @return the learning rate (A.K.A step size/length)
	 */
	virtual float64_t get_learning_rate()
	{
		m_call_counter++;
		return m_initial_learning_rate/CMath::pow(m_intercept+m_slope*m_call_counter,m_exponent);
	}

	/** Set the initial learning rate
	 *
	 * @param learning_rate learning_rate must be positive
	 */
	virtual void set_initial_learning_rate(float64_t initial_learning_rate)
	{
		REQUIRE(initial_learning_rate>0.0, "Initial learning rate (%f) should be positive\n",
			initial_learning_rate);
		m_initial_learning_rate=initial_learning_rate;
	}

	/** Set the exponent term
	 *
	 * @param exponent exponent term should be positive
	 */
	virtual void set_exponent(float64_t exponent)
	{
		REQUIRE(exponent>0.0, "Exponent (%f) should be positive\n", exponent);
		m_exponent=exponent;
	}

	/** Set the slope term
	 *
	 * @param slope slope term should be positive
	 */
	virtual void set_slope(float64_t slope)
	{
		REQUIRE(slope>0.0,"Slope (%f) should be positive\n", slope);
		m_slope=slope;
	}

	/** Set the intercept term
	 *
	 * @param intercept intercept term should be positive
	 */
	virtual void set_intercept(float64_t intercept)
	{
		REQUIRE(intercept>=0, "Intercept (%f) should be non-negative\n",
			intercept);
		m_intercept=intercept;
	}

	/** Reset the call counter
	 * Afer this method is called, call counter will become 0.
	 */
	virtual void reset_call_counter()
	{
		m_call_counter=0;
	}

	/** Update a context object to store mutable variables
	 *
	 * This method will be called by
	 * DescendUpdaterWithCorrection::update_context()
	 *
	 * @param context, a context object
	 */
	virtual void update_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
		std::string key="InverseScalingLearningRate::m_call_counter";
		context->save_data(key, m_call_counter);
	}

	/** Return a context object which stores mutable variables
	 * Usually it is used in serialization.
	 *
	 * This method will be called by
	 * DescendUpdaterWithCorrection::load_from_context(CMinimizerContext* context)
	 *
	 * @return a context object
	 */
	virtual void load_from_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
		std::string key="InverseScalingLearningRate::m_call_counter";
		m_call_counter=context->get_data_int32(key);
	}
protected:
	/*  counter */
	int32_t m_call_counter;
	/*  exponent */
	float64_t m_exponent;
	/*  slope */
	float64_t m_slope;
	/*  intercept */
	float64_t m_intercept;
	/*  init_learning_rate */
	float64_t m_initial_learning_rate;
private:
	/*  Init */
	void init()
	{
		m_call_counter=0;
		m_exponent=0.5;
		m_initial_learning_rate=1.0;
		m_intercept=0.0;
		m_slope=1.0;
	}
};

}

#endif

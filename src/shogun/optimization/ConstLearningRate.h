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

#ifndef CONSTLEARNINGRATE_H
#define CONSTLEARNINGRATE_H
#include <shogun/optimization/LearningRate.h>

namespace shogun
{
/** @brief This implements the const learning rate class for a descent-based minimizer.
 *
 * This class gives a const learning rate during descent update.
 *
 */

class ConstLearningRate: public LearningRate
{
public:
	/**  Constructor */
	ConstLearningRate():LearningRate() { init(); }

	/**  Destructor */
	~ConstLearningRate() override {}

	/** returns the name of the class
	 *
	 * @return name ConstLearningRate
	 */
	const char* get_name() const override { return "ConstLearningRate"; }


	/** Set the const learning rate
	 *
	 * @param learning_rate learning_rate must be positive and usually is not greater than 1.0
	 */
	virtual void set_const_learning_rate(float64_t learning_rate);

	/** Get the learning rate for descent direction
	 * Note that the learning rate usually is positive
	 *
	 * @param iter_counter the number of iterations
	 *
	 *
	 * @return the learning rate (A.K.A step size/length)
	 */
	float64_t get_learning_rate(int32_t iter_counter) override;

protected:

	/** const_learning_rate */
	float64_t m_const_learning_rate;

private:
	/**  Init */
	void init();
};

}

#endif

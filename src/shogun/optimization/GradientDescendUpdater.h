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

#ifndef GRADIENTDESCENDUPDATER_H
#define GRADIENTDESCENDUPDATER_H
#include <shogun/optimization/DescendUpdater.h>
#include <shogun/optimization/LearningRate.h>
namespace shogun
{
/** @brief The class is about the plain gradient descend method.
 *
 * Given a target variable, \f$w\f$, and its gradient, \f$d\f$, this class performs the following update.
 * \f[
 *  w^{new} = w - \lambda d
 * \f],
 * where \f$\lambda\f$ is a learning rate.
 */
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class GradientDescendUpdater: public DescendUpdater
{
public:
	/* Constructor */
	GradientDescendUpdater();

	/** Constructor
	 * @param learning_rate a learning_rate class
	 */
	GradientDescendUpdater(LearningRate *learning_rate);

	/* Destructor */
	virtual ~GradientDescendUpdater();

	/** Set learning rate
	 *
	 * @param learning_rate a learning_rate class
	 */
	virtual void set_learning_rate(LearningRate *learning_rate);

	/** Update the variable_reference based on the negative_descend_direction
	 *
	 * Plain gradient descend update:
	 * variable_reference -= learning_rate*gradient
	 * 
	 * @param variable_reference a reference of the target variable, which supports in place update
	 * @param gradient gradient of the variable_reference
	 */
	virtual void update_variable(SGVector<float64_t> variable_reference, SGVector<float64_t> gradient);

	/** Update a context object to store mutable variables
	 *
	 * @param context, a context object
	 */
	virtual void update_context(CMinimizerContext* context);

	/** Return a context object which stores mutable variables
	 * Usually it is used in serialization.
	 *
	 * @return a context object
	 */
	virtual void load_from_context(CMinimizerContext* context);

protected:

	/* learning_rate */
	LearningRate* m_learning_rate;

private:
	/*  init */
	void init();
};

}
#endif

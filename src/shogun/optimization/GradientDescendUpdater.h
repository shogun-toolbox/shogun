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
#include <shogun/optimization/DescendUpdaterWithCorrection.h>
#include <shogun/optimization/LearningRate.h>
namespace shogun
{
/** @brief The class implements the gradient descend method.
 *
 * Given a target variable, \f$w\f$, and its gradient, \f$d\f$,
 * without gradient correction (eg, momentum correction),
 * this class performs the following update.
 * \f[
 *  w^{new} = w - \lambda d
 * \f],
 * where \f$\lambda\f$ is a learning rate.
 */
class GradientDescendUpdater: public DescendUpdaterWithCorrection
{
public:
	/* Constructor */
	GradientDescendUpdater();

	/* Destructor */
	virtual ~GradientDescendUpdater();

protected:
	/** Get the negative descend direction given current variable and gradient
	 *
	 * It will be called at update_variable()
	 *
	 * @param variable current variable
	 * @param gradient current gradient
	 * @param idx the index of the variable
	 * @param learning_rate learning rate
	 * 
	 * @return negative descend direction (that is, the given gradient in the class)
	 */
	virtual float64_t get_negative_descend_direction(float64_t variable,
		float64_t gradient, index_t idx, float64_t learning_rate);

private:
	/*  Init */
	void init();
};

}
#endif

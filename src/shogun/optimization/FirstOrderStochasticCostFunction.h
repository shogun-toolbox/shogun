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

#ifndef FIRSTORDERSTOCHASTICCOSTFUNCTION_H
#define FIRSTORDERSTOCHASTICCOSTFUNCTION_H
#include <shogun/lib/config.h>
#include <shogun/optimization/FirstOrderCostFunction.h>
namespace shogun
{
/** @brief The first order stochastic cost function base class.
 *
 * The class gives the interface used in first order stochastic minimizers
 *
 * the cost function must be Written as a finite sample-specific sum of cost  
 * for example:
 * f=0.5*\sum_i{ (y_i-x_i'w)^2 }, where (y_i,x_i) is the i-th sample, y_i is the label and x_i is the features 
 *
 */
class CFirstOrderStochasticCostFunction: public CFirstOrderCostFunction
{
public:
	/* initialize to get samples
	 *
	 */
	virtual void begin_sample()=0;

	/* get next sample
	 *
	 * @return false if reach the end of sample sequence
	 * */
	virtual bool next_sample()=0;
};

}

#endif

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

#ifndef FIRSTORDERSAGCOSTFUNCTION_H
#define FIRSTORDERSAGCOSTFUNCTION_H
#include <shogun/lib/config.h>
#include <shogun/optimization/FirstOrderStochasticCostFunction.h>
namespace shogun
{
/** @brief The class is about a stochastic cost function for stochastic average minimizers.
 *
 * The cost function must be written as a finite sample-specific sum of cost.  
 * For example, least squares cost function,
 * \f[
 * f(w)=\frac{ \sum_i^n{ (y_i-w^T x_i)^2 } }{2}
 * \f]
 * where \f$n\f$ is the sample size,
 * \f$(y_i,x_i)\f$ is the i-th sample,
 * \f$y_i\f$ is the label and \f$x_i\f$ is the features 
 *
 * A stochastic average minimizer uses average sample gradients ( get_average_gradient() )
 * to reduce variance related to stochastic gradients.
 *
 * Well known stochastic average methods are:
 * SVRG, Johnson, Rie, and Tong Zhang.
 * "Accelerating stochastic gradient descent using predictive variance reduction."
 * Advances in Neural Information Processing Systems. 2013.
 *
 * SAG, Schmidt, Mark, Nicolas Le Roux, and Francis Bach.
 * "Minimizing finite sums with the stochastic average gradient."
 * arXiv preprint arXiv:1309.2388 (2013).
 *
 * SAGA, Defazio, Aaron, Francis Bach, and Simon Lacoste-Julien.
 * "SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives."
 * Advances in Neural Information Processing Systems. 2014.
 *
 * SDCA, Shalev-Shwartz, Shai, and Tong Zhang.
 * "Stochastic dual coordinate ascent methods for regularized loss."
 * The Journal of Machine Learning Research 14.1 (2013): 567-599.
 *
 */
class FirstOrderSAGCostFunction
	: public FirstOrderStochasticCostFunction
{
public:

	/** Get the sample size 
	 *
	 * @return the sample size
	 */
	virtual int32_t get_sample_size()=0;

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
	virtual SGVector<float64_t> get_average_gradient()=0;

	/** Get the SAMPLE gradient value wrt target variables 
	 *
	 * WARNING
	 * This method does return 
	 * \f$ \frac{\partial f_i(w) }{\partial w} \f$
	 * instead of
	 * \f$\sum_i^n{ \frac{\partial f_i(w) }{\partial w} }\f$
	 *
	 * For least squares cost function, that is the value of
	 * \f$\frac{\partial f_i(w) }{\partial w}\f$ given \f$w\f$ is known
	 * where the index \f$i\f$ is obtained by next_sample() 
	 *
	 * @return sample gradient of target variables
	 */
	virtual SGVector<float64_t> get_gradient()=0;

	/** Get the cost given current target variables 
	 *
	 * For least squares cost function, that is the value of \f$f(w)\f$.
	 *
	 * @return cost
	 */
	virtual float64_t get_cost()=0;
};

}

#endif

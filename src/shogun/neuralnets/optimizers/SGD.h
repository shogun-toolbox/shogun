/*
 * Copyright (c) 2015, Shogun Toolbox Foundation
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its 
 * contributors may be used to endorse or promote products derived from this 
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * Written (W) 2015 Khaled Nasr, Sergey Lisitsyn
 */

#ifndef __SGDNEURALNETWORKOPTIMIZER_H__
#define __SGDNEURALNETWORKOPTIMIZER_H__

#include <shogun/neuralnets/NeuralNetworkOptimizer.h>

namespace shogun
{

class CSGDNeuralNetworkOptimizer : public CNeuralNetworkOptimizer
{
public:
	CSGDNeuralNetworkOptimizer();

	virtual bool optimize(CNeuralNetwork* network,
			CDotFeatures* data, SGMatrix<float64_t> targets);

	virtual const char* get_name() const { return "SGDNeuralNetworkOptimizer"; }

private:

	void prepare();

public:

	/** size of the mini-batch used during gradient descent training, 
	 * if 0 full-batch training is performed
	 * default value is 0
	 */
	int32_t gd_mini_batch_size;
	
	/** gradient descent learning rate, defualt value 0.1 */
	float64_t gd_learning_rate;
	
	/** gradient descent learning rate decay
	 * learning rate is updated at each iteration i according to: 
	 * alpha(i)=decay*alpha(i-1)
	 * default value is 1.0 (no decay)
	 */
	float64_t gd_learning_rate_decay;
	
	/** gradient descent momentum multiplier
	 * 
	 * default value is 0.9
	 * 
	 * For more details on momentum, see this 
	 * [paper](http://jmlr.org/proceedings/papers/v28/sutskever13.html) 
	 * [Sutskever, 2013]
	 */
	float64_t gd_momentum;
	
	/** Used to damp the error fluctuations when stochastic gradient descent is 
	 * used. damping is done according to: 
	 * error_damped(i) = c*error(i) + (1-c)*error_damped(i-1)
	 * where c is the damping coefficient
	 * 
	 * If -1, the damping coefficient is automatically computed according to:
	 * c = 0.99*gd_mini_batch_size/training_set_size + 1e-2;
	 * 
	 * default value is -1
	 */
	float64_t gd_error_damping_coeff;
};

}

#endif

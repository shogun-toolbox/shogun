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

#ifndef __LBFGSNEURALNETWORKOPTIMIZER_H__
#define __LBFGSNEURALNETWORKOPTIMIZER_H__

#include <shogun/neuralnets/NeuralNetworkOptimizer.h>

namespace shogun
{

class CLBFGSNeuralNetworkOptimizer : public CNeuralNetworkOptimizer
{
public:
	CLBFGSNeuralNetworkOptimizer();

	virtual bool optimize(CNeuralNetwork* network,
			CDotFeatures* data, SGMatrix<float64_t> targets);
	
	virtual const char* get_name() const { return "LBFGSNeuralNetworkOptimizer"; }

private:
	/** callback for l-bfgs */
	static float64_t lbfgs_evaluate(void *userdata, 
			const float64_t *W, 
			float64_t *grad, 
			const int32_t n, 
			const float64_t step);

	/** callback for l-bfgs */
	static int lbfgs_progress(void *instance,
			const float64_t *x,
			const float64_t *g,
			const float64_t fx,
			const float64_t xnorm,
			const float64_t gnorm,
			const float64_t step,
			int n,
			int k,
			int ls
			);
};

}

#endif

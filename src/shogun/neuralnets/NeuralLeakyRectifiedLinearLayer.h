/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
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
 * Written (W) 2015 Sanuj Sharma
 */

#ifndef __NEURALLEAKYRECTIFIEDLINEARLAYER_H__
#define __NEURALLEAKYRECTIFIEDLINEARLAYER_H__

#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>

namespace shogun
{
/** @brief Neural layer with [leaky rectified linear neurons]
 * (http://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29).
 *
 * Activations are computed according to max(alpha*(W*x+b),W*x+b) where W is the weight
 * matrix, b is the bias vector, x is the input vector, and alpha is a parameter usually
 * between 0 and 1. Default value of alpha is 0.01.
 *
 * When used as an output layer, a squared error measure is used
 */
class CNeuralLeakyRectifiedLinearLayer : public CNeuralRectifiedLinearLayer
{
public:
	/** default constructor */
	CNeuralLeakyRectifiedLinearLayer();

	/** Constuctor
	 *
	 * @param num_neurons Number of neurons in this layer
	 */
	CNeuralLeakyRectifiedLinearLayer(int32_t num_neurons);

	virtual ~CNeuralLeakyRectifiedLinearLayer() {}

	/** Sets the value of alpha used to calculate max(alpha*(W*x+b),W*x+b)
	 *
	 * @param alpha new value of alpha
	 */
	virtual void set_alpha(float64_t alpha) { m_alpha=alpha; }

	/** Gets the value of alpha used to calculate max(alpha*(W*x+b),W*x+b)
	 *
	 * @return alpha
	 */
	virtual float64_t get_alpha() { return m_alpha; }

	/** Computes the activations of the neurons in this layer, results should
	 * be stored in m_activations. To be used only with non-input layers
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 */
	virtual void compute_activations(SGVector<float64_t> parameters,
		CDynamicObjectArray* layers);

	virtual const char* get_name() const { return "NeuralLeakyRectifiedLinearLayer"; }

protected:
	/** Parameter used to calculate max(alpha*(W*x+b),W*x+b).
	 * Default value is 0.01
	 */
	float64_t m_alpha;
};

}
#endif //__NEURALLEAKYRECTIFIEDLINEARLAYER_H__

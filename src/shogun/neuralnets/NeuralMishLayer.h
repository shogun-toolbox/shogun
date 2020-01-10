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
 * Written (W) 2020 Manjunath Bhat
 */

#ifndef __NEURALMISHLAYER_H__
#define __NEURALMISHLAYER_H__

#include <shogun/neuralnets/NeuralLinearLayer.h>

namespace shogun
{
/** @brief Neural layer with [Mish activated neurons]
 * (https://arxiv.org/abs/1908.08681).
 *
 * Activations are computed according to x*tanh(log(1 + e^x))
 *
 * When used as an output layer, a squared error measure is used
 */
class NeuralMishLayer : public NeuralLinearLayer
{
public:
	/** default constructor */
	NeuralMishLayer();

	/** Constuctor
	 *
	 * @param num_neurons Number of neurons in this layer
	 */
	NeuralMishLayer(int32_t num_neurons);

	virtual ~NeuralMishLayer() {}

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
	virtual void compute_activations(
		SGVector<float64_t> parameters,
		const std::vector<std::shared_ptr<NeuralLayer>>& layers);

	virtual const char* get_name() const { return "NeuralMishLayer"; }
};

}
#endif //__NEURALMISHLAYER_H__

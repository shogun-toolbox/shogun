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

#include <shogun/neuralnets/NeuralLeakyRectifiedLinearLayer.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

CNeuralLeakyRectifiedLinearLayer::CNeuralLeakyRectifiedLinearLayer() : CNeuralRectifiedLinearLayer()
{
	m_alpha=0.01;
}

CNeuralLeakyRectifiedLinearLayer::CNeuralLeakyRectifiedLinearLayer(int32_t num_neurons):
CNeuralRectifiedLinearLayer(num_neurons)
{
	m_alpha=0.01;
}

void CNeuralLeakyRectifiedLinearLayer::compute_activations(
	SGVector<float64_t> parameters,
	CDynamicObjectArray* layers)
{
	CNeuralLinearLayer::compute_activations(parameters, layers);

	int32_t len = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<len; i++)
	{
		m_activations[i] = CMath::max<float64_t>(m_alpha*m_activations[i], m_activations[i]);
	}
}
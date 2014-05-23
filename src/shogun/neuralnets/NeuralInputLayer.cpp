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
 * Written (W) 2014 Khaled Nasr
 */

#include <shogun/neuralnets/NeuralInputLayer.h>

using namespace shogun;

CNeuralInputLayer::CNeuralInputLayer() : CNeuralLayer()
{
	init();
}

CNeuralInputLayer::CNeuralInputLayer(int32_t num_neurons, int32_t start_index): 
CNeuralLayer(num_neurons)
{
	init();
	m_start_index = start_index;
}

void CNeuralInputLayer::compute_activations(SGMatrix< float64_t > inputs)
{
	if (m_start_index == 0)
	{
		memcpy(m_activations.matrix, inputs.matrix, 
			m_num_neurons*m_batch_size*sizeof(float64_t));
	}
	else
	{
		for (int32_t i=0; i<m_num_neurons; i++)
			for (int32_t j=0; j<m_batch_size; j++)
				m_activations(i,j) = inputs(m_start_index+i, j);
	}
}

void CNeuralInputLayer::init()
{
	m_start_index = 0;
	SG_ADD(&m_start_index, "start_index",
	       "Start Index", MS_NOT_AVAILABLE);
}

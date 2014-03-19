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

#include <shogun/base/Parameter.h>
#include <shogun/neuralnets/NeuralLayer.h>

using namespace shogun;

CNeuralLayer::CNeuralLayer() 
: CSGObject(), m_num_neurons(0), m_previous_layer_num_neurons(0),
	m_batch_size(0)
{
	init();
}


CNeuralLayer::CNeuralLayer(int32_t num_neurons)
: CSGObject(), m_num_neurons(num_neurons), 
	m_previous_layer_num_neurons(0), m_batch_size(0)
{
	init();
}

CNeuralLayer::~CNeuralLayer()
{
}

void CNeuralLayer::initialize(int32_t previous_layer_num_neurons)
{
	m_previous_layer_num_neurons = previous_layer_num_neurons;
}

void CNeuralLayer::set_batch_size(int32_t batch_size)
{
	m_batch_size = batch_size;
	
	if (m_activations.vector!=NULL) SG_FREE(m_activations.vector);
	if (m_input_gradients.vector!=NULL) SG_FREE(m_input_gradients.vector);
	if (m_local_gradients.vector!=NULL) SG_FREE(m_local_gradients.vector);
	
	m_activations.vlen = m_num_neurons * m_batch_size;
	m_input_gradients.vlen = m_previous_layer_num_neurons * m_batch_size;
	m_local_gradients.vlen = m_num_neurons * m_batch_size;
	
	m_activations.vector = SG_MALLOC(float64_t, m_activations.vlen);
	m_input_gradients.vector = SG_MALLOC(float64_t, m_input_gradients.vlen);
	m_local_gradients.vector = SG_MALLOC(float64_t, m_local_gradients.vlen);
}

void CNeuralLayer::init()
{
	SG_ADD(&m_num_neurons, "num_neurons",
	       "Number of Neurons", MS_NOT_AVAILABLE);
	SG_ADD(&m_previous_layer_num_neurons, "previous_layer_num_neurons",
	       "Number of neurons in the previous layer", MS_NOT_AVAILABLE);
	SG_ADD(&m_batch_size, "batch_size",
	       "Batch Size", MS_NOT_AVAILABLE);
	SG_ADD(&m_activations, "activations",
	       "Activations", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_gradients, "input_gradients",
	       "Input Gradients", MS_NOT_AVAILABLE);
	SG_ADD(&m_local_gradients, "local_gradients",
	       "Local Gradients", MS_NOT_AVAILABLE);
}

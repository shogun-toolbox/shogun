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
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CNeuralLayer::CNeuralLayer()
: CSGObject()
{
	init();
}


CNeuralLayer::CNeuralLayer(int32_t num_neurons)
: CSGObject()
{
	init();
	m_num_neurons = num_neurons;
	m_width = m_num_neurons;
	m_height = 1;
}

CNeuralLayer::~CNeuralLayer()
{
}

void CNeuralLayer::initialize(CDynamicObjectArray* layers,
		SGVector< int32_t > input_indices)
{
	m_input_indices = input_indices;
	m_input_sizes = SGVector<int32_t>(input_indices.vlen);

	for (int32_t i=0; i<m_input_sizes.vlen; i++)
	{
		CNeuralLayer* layer = (CNeuralLayer*)layers->element(m_input_indices[i]);
		m_input_sizes[i] = layer->get_num_neurons();
		SG_UNREF(layer);
	}
}

void CNeuralLayer::set_batch_size(int32_t batch_size)
{
	m_batch_size = batch_size;

	m_activations = SGMatrix<float64_t>(m_num_neurons, m_batch_size);
	m_dropout_mask = SGMatrix<bool>(m_num_neurons, m_batch_size);

	if (!is_input())
	{
		m_activation_gradients =
			SGMatrix<float64_t>(m_num_neurons, m_batch_size);
		m_local_gradients = SGMatrix<float64_t>(m_num_neurons, m_batch_size);
	}
}

void CNeuralLayer::dropout_activations()
{
	if (dropout_prop==0.0) return;

	if (is_training)
	{
		int32_t len = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<len; i++)
		{
			m_dropout_mask[i] = CMath::random(0.0,1.0) >= dropout_prop;
			m_activations[i] *= m_dropout_mask[i];
		}
	}
	else
	{
		int32_t len = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<len; i++)
			m_activations[i] *= (1.0-dropout_prop);
	}
}

void CNeuralLayer::init()
{
	m_num_neurons = 0;
	m_width = 0;
	m_height = 0;
	m_num_parameters = 0;
	m_batch_size = 0;
	dropout_prop = 0.0;
	contraction_coefficient = 0.0;
	is_training = false;
	autoencoder_position = NLAP_NONE;

	SG_ADD(&m_num_neurons, "num_neurons",
	       "Number of Neurons", MS_NOT_AVAILABLE);
	SG_ADD(&m_width, "width",
	       "Width", MS_NOT_AVAILABLE);
	SG_ADD(&m_height, "height",
	       "Height", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_indices, "input_indices",
	       "Input Indices", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_sizes, "input_sizes",
	       "Input Sizes", MS_NOT_AVAILABLE);
	SG_ADD(&dropout_prop, "dropout_prop",
	       "Dropout Probabilty", MS_NOT_AVAILABLE);
	SG_ADD(&contraction_coefficient, "contraction_coefficient",
	       "Contraction Coefficient", MS_NOT_AVAILABLE);
	SG_ADD(&is_training, "is_training",
	       "is_training", MS_NOT_AVAILABLE);
	SG_ADD(&m_batch_size, "batch_size",
	       "Batch Size", MS_NOT_AVAILABLE);
	SG_ADD(&m_activations, "activations",
	       "Activations", MS_NOT_AVAILABLE);
	SG_ADD(&m_activation_gradients, "activation_gradients",
	       "Activation Gradients", MS_NOT_AVAILABLE);
	SG_ADD(&m_local_gradients, "local_gradients",
	       "Local Gradients", MS_NOT_AVAILABLE);
	SG_ADD(&m_dropout_mask, "dropout_mask",
	       "Dropout mask", MS_NOT_AVAILABLE);

	SG_ADD((machine_int_t*)&autoencoder_position, "autoencoder_position",
	       "Autoencoder Position", MS_NOT_AVAILABLE);
}

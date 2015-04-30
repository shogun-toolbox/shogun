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

#include <shogun/neuralnets/NeuralLayers.h>

#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLinearLayer.h>
#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralLeakyRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/neuralnets/NeuralSoftmaxLayer.h>

using namespace shogun;

CNeuralLayers::CNeuralLayers() : CSGObject(), m_layers(new CDynamicObjectArray)
{
}

CNeuralLayers::~CNeuralLayers()
{
	SG_UNREF(m_layers)
}

CNeuralLayers* CNeuralLayers::input(int32_t size)
{
	return with_layer(new CNeuralInputLayer(size));
}

CNeuralLayers* CNeuralLayers::logistic(int32_t size)
{
	return with_layer(new CNeuralLogisticLayer(size));
}

CNeuralLayers* CNeuralLayers::linear(int32_t size)
{
	return with_layer(new CNeuralLinearLayer(size));
}

CNeuralLayers* CNeuralLayers::rectified_linear(int32_t size)
{
	return with_layer(new CNeuralRectifiedLinearLayer(size));
}

CNeuralLayers* CNeuralLayers::leaky_rectified_linear(int32_t size)
{
	return with_layer(new CNeuralLeakyRectifiedLinearLayer(size));
}

CNeuralLayers* CNeuralLayers::softmax(int32_t size)
{
	return with_layer(new CNeuralSoftmaxLayer(size));
}

CNeuralLayers* CNeuralLayers::with_layer(CNeuralLayer* layer)
{
	m_layers->push_back(layer);
	return this;
}

CDynamicObjectArray* CNeuralLayers::done()
{
	SG_REF(m_layers);
	return m_layers;
}

void CNeuralLayers::clear()
{
	m_layers->clear_array();
}

bool CNeuralLayers::empty()
{
	return (m_layers->get_array_size() == 0);
}

const char* CNeuralLayers::get_name() const
{
	return "NeuralLayers";
}

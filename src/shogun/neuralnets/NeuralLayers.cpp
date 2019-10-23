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

NeuralLayers::NeuralLayers() : SGObject()
{
}

NeuralLayers::~NeuralLayers()
{

}

std::shared_ptr<NeuralLayers> NeuralLayers::input(int32_t size)
{
	return with_layer(std::make_shared<NeuralInputLayer>(size));
}

std::shared_ptr<NeuralLayers> NeuralLayers::logistic(int32_t size)
{
	return with_layer(std::make_shared<NeuralLogisticLayer>(size));
}

std::shared_ptr<NeuralLayers> NeuralLayers::linear(int32_t size)
{
	return with_layer(std::make_shared<NeuralLinearLayer>(size));
}

std::shared_ptr<NeuralLayers> NeuralLayers::rectified_linear(int32_t size)
{
	return with_layer(std::make_shared<NeuralRectifiedLinearLayer>(size));
}

std::shared_ptr<NeuralLayers> NeuralLayers::leaky_rectified_linear(int32_t size)
{
	return with_layer(std::make_shared<NeuralLeakyRectifiedLinearLayer>(size));
}

std::shared_ptr<NeuralLayers> NeuralLayers::softmax(int32_t size)
{
	return with_layer(std::make_shared<NeuralSoftmaxLayer>(size));
}

std::shared_ptr<NeuralLayers> NeuralLayers::with_layer(const std::shared_ptr<NeuralLayer>& layer)
{
	m_layers.push_back(layer);
	return shared_from_this()->as<NeuralLayers>();
}

const std::vector<std::shared_ptr<NeuralLayer>>& NeuralLayers::done()
{

	return m_layers;
}

void NeuralLayers::clear()
{
	m_layers.clear();
}

bool NeuralLayers::empty()
{
	return m_layers.empty();
}

const char* NeuralLayers::get_name() const
{
	return "NeuralLayers";
}

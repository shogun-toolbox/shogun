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
#include <shogun/features/DotFeatures.h>

using namespace shogun;

CNeuralInputLayer::CNeuralInputLayer() : CNeuralLayer()
{
	init();
}

CNeuralInputLayer::CNeuralInputLayer(int32_t num_neurons): 
CNeuralLayer(num_neurons)
{
	init();
}

CNeuralInputLayer::CNeuralInputLayer(int32_t width, int32_t height, 
	int32_t num_channels): CNeuralLayer(width*height*num_channels)
{
	init();
	m_width = width;
	m_height = height;
}

void CNeuralInputLayer::compute_activations(CFeatures* data)
{
	CDotFeatures* dot_data = dynamic_cast<CDotFeatures*>(data);
	REQUIRE(dot_data, "Dot features expected");
	m_activations.zero();

	int32_t dim = dot_data->get_dim_feature_space();
	REQUIRE(m_activations.num_rows == dim, "%d should match dimensionality %d", m_activations.num_rows, dim);
	REQUIRE(m_activations.num_cols == data->get_num_vectors(), "%d should match number of vectors %d", m_activations.num_rows, data->get_num_vectors());
	for (int i=0; i<data->get_num_vectors(); i++)
		dot_data->add_to_dense_vec(1.0, i, m_activations.matrix + dim*i, dim);

	if (gaussian_noise > 0)
	{
		int32_t len = m_num_neurons*m_batch_size;
		for (int32_t k=0; k<len; k++)
			m_activations[k] += CMath::normal_random(0.0, gaussian_noise);
	}
}

void CNeuralInputLayer::init()
{
	gaussian_noise = 0;
	SG_ADD(&gaussian_noise, "gaussian_noise",
	       "Gaussian Noise Standard Deviation", MS_NOT_AVAILABLE);
}

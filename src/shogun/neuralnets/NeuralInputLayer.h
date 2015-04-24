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

#ifndef __NEURALINPUTLAYER_H__
#define __NEURALINPUTLAYER_H__

#include <shogun/lib/common.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief Represents an input layer
 */
class CNeuralInputLayer : public CNeuralLayer
{
public:
	/** default constructor */
	CNeuralInputLayer();
	
	/** Constuctor
	 * 
	 * @param num_neurons Number of neurons in this layer
	 * 
	 */
	CNeuralInputLayer(int32_t num_neurons);
	
	/** Constructs an input layer that deals with images (for convolutional nets).
	 * Sets the number of neurons to width*height*num_channels
	 * 
	 * @param width Width of the image
	 * 
	 * @param height Width of the image
	 * 
	 * @param num_channels Number of channels
	 * 
	 */
	CNeuralInputLayer(int32_t width, int32_t height, int32_t num_channels);
	
	virtual ~CNeuralInputLayer() {}
	
	/** Returns true */
	virtual bool is_input() { return true; }
	
	/** Copies features from inputs into the 
	 * layer's activations
	 * 
	 * @param inputs Input features matrix, size num_features*num_cases
	 */
	virtual void compute_activations(CFeatures* inputs);

	virtual const char* get_name() const { return "NeuralInputLayer"; }
	
private:
	void init();

public:
	/** Standard deviation of the gaussian noise added to the activations of 
	 * the layer. Useful for denoising autoencoders. Default value is 0.0.
	 */
	float64_t gaussian_noise;

};
}
#endif

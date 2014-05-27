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

namespace shogun
{
/** @brief Represents an input layer. The layer can be either 
 * connected to all the input features that a network receives (default) or 
 * connected to just a small part of those features
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
	 * @param start_index Index of the first feature that the layer connects to, 
	 * i.e the activations of the layer are copied from 
	 * input_features[start_index:start_index+num_neurons]
	 */
	CNeuralInputLayer(int32_t num_neurons, int32_t start_index = 0);
	
	virtual ~CNeuralInputLayer() {}
	
	/** Returns true */
	virtual bool is_input() { return true; }
	
	/** Copies inputs[start_index:start_index+num_neurons, :] into the 
	 * layer's activations
	 * 
	 * @param inputs Input features matrix, size num_features*num_cases
	 */
	virtual void compute_activations(SGMatrix<float64_t> inputs);
	
	/** Gets the index of the first feature that the layer connects to, 
	 * i.e the activations of the layer are copied from 
	 * input_features[start_index:start_index+num_neurons]
	 */
	virtual int32_t get_start_index() { return m_start_index; }
	
	/** Sets the index of the first feature that the layer connects to, 
	 * i.e the activations of the layer are copied from 
	 * input_features[start_index:start_index+num_neurons]
	 */
	virtual void set_start_index(int32_t i) { m_start_index = i; }
	
	virtual const char* get_name() const { return "NeuralInputLayer"; }
	
private:
	void init();

protected:
	/** Index of the first feature that the layer connects to, 
	 * i.e the activations of the layer are copied from 
	 * input_features[start_index:start_index+num_neurons]
	 */
	int32_t m_start_index;
};
}
#endif

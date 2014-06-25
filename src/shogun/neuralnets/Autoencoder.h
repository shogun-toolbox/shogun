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

#ifndef __AUTOENCODER_H__
#define __AUTOENCODER_H__

#include <shogun/lib/common.h>
#include <shogun/neuralnets/NeuralNetwork.h>

namespace shogun
{
class CNeuralLayer;
template <class T> class CDenseFeatures;

enum EAENoiseType
{
	AENT_NONE=0,
	AENT_DROPOUT=1,
	AENT_GAUSSIAN=2
};	

/** @brief Represents a single layer neural autoencoder
 * 
 * An [autoencoder](http://deeplearning.net/tutorial/dA.html#autoencoders) is 
 * a neural network that has three layers: an input layer, a hidden (encoding) 
 * layer, and a decoding layer. The network is trained to reconstruct its inputs,
 * which forces the hidden layer to try to learn good representations of the 
 * inputs.
 * 
 * This class supports training normal autoencoders and 
 * [denoising autoencoders](http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/217)
 * [Vincent, 2008]. To use denoising autoencoders set noise_type and noise_parameter 
 * to specify the type and strength of the noise.
 */
class CAutoencoder : public CNeuralNetwork
{
public:
	/** default constructor */
	CAutoencoder();
	
	/** Constructor
	 * 
	 * @param num_inputs Number of inputs
	 * @param hidden_layer Hidden layer. Can be any CNeuralLayer based object 
	 * that supports being used as a hidden layer
	 * @param decoding_layer Decoding layer. Must have the same number of neurons 
	 * as num_inputs. Can be any CNeuralLayer based object that supports being 
	 * used as an output layer. If NULL, a CNeuralLinearLayer is used.
	 */
	CAutoencoder(int32_t num_inputs, CNeuralLayer* hidden_layer, 
		CNeuralLayer* decoding_layer=NULL);
	
	/** Trains the autoencoder 
	 * 
	 * @param data Training examples
	 * 
	 * @return True if training succeeded, false otherwise
	 */
	virtual bool train(CFeatures* data);
	
	/** Computes the activation of the hidden layer given the input data
	 * 
	 * @param data Input features
	 * 
	 * @return Transformed features
	 */
	virtual CDenseFeatures<float64_t>* transform(
		CDenseFeatures<float64_t>* data);
	
	/** Reconstructs the input data
	 * 
	 * @param data Input features
	 * 
	 * @return Reconstructed features
	 */
	virtual CDenseFeatures<float64_t>* reconstruct(
		CDenseFeatures<float64_t>* data);
	
	virtual ~CAutoencoder() {}
	
	virtual const char* get_name() const { return "Autoencoder"; }
	
private:
	void init();
	
public:
	/** Noise type for denoising autoencoders. 
	 * 
	 * If set to AENT_DROPOUT, inputs are randomly set to zero during each 
	 * iteration of training with probability noise_parameter.
	 * 
	 * If set to AENT_GAUSSIAN, gaussian noise with zero mean and noise_parameter 
	 * standard deviation is added to the inputs.
	 * 
	 * Default value is AENT_NONE
	 */
	EAENoiseType noise_type;
	
	/** Controls the strength of the noise, depending on noise_type */
	float64_t noise_parameter;
};
}
#endif

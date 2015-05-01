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
#include <shogun/neuralnets/NeuralLayer.h>

namespace shogun
{
template <class T> class CDenseFeatures;
class CNeuralConvolutionalLayer;

/** @brief Determines the noise type for denoising autoencoders */
enum EAENoiseType
{
	/** No noise is applied */
	AENT_NONE=0,

	/** Each neuron in the input layer is randomly set to zero with some probability */
	AENT_DROPOUT=1,

	/** Gaussian noise is added to neurons in the input layer */
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
 *
 * NOTE: LBFGS does not work properly with denoising autoencoders due to their
 * stochastic nature. Use gradient descent instead.
 *
 * [Contractive autoencoders](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Rifai_455.pdf)
 * [Rifai, 2011] are also supported. To use them, call set_contraction_coefficient().
 * Denoising can also be used with contractive autoencoders through noise_type
 * and noise_parameter.
 *
 * [Convolutional autoencoders](http://www.idsia.ch/~ciresan/data/icann2011.pdf)
 * [J Masci, 2011] are also supported. Simply build the autoencoder
 * using CNeuralConvolutionalLayer objects.
 *
 * NOTE: Contractive convolutional autoencoders are not supported.
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
	 * @param sigma Standard deviation of the gaussian used to initialize the
	 * parameters
	 */
	CAutoencoder(int32_t num_inputs, CNeuralLayer* hidden_layer,
		CNeuralLayer* decoding_layer=NULL, float64_t sigma = 0.01);

	/** Constructor for convolutional autoencoders
	 *
	 * @param input_width Width of the input images
	 * @param input_height height of the input images
	 * @param input_num_channels number of channels in the input images
	 * @param hidden_layer Hidden layer
	 * @param decoding_layer Decoding layer. Should have the same dimensions as
	 * the inputs.
	 * @param sigma Standard deviation of the gaussian used to initialize the
	 * parameters
	 */
	CAutoencoder(int32_t input_width, int32_t input_height, int32_t input_num_channels,
		CNeuralConvolutionalLayer* hidden_layer,
		CNeuralConvolutionalLayer* decoding_layer, float64_t sigma = 0.01);

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

	/** Sets the contraction coefficient
	 *
	 * For contractive autoencoders [Rifai, 2011], a term:
	 * \f[ \frac{\lambda}{N} \sum_{k=0}^{N-1} \left \| J(x_k) \right \|^2_F \f]
	 * is added to the error, where \f$ \left \| J(x_k)) \right \|^2_F \f$ is the
	 * Frobenius norm of the Jacobian of the activations of the hidden layer
	 * with respect to its inputs, \f$ N \f$ is the batch size, and
	 * \f$ \lambda \f$ is the contraction coefficient.
	 *
	 * @param coeff Contraction coefficient
	 */
	virtual void set_contraction_coefficient(float64_t coeff)
	{
		m_contraction_coefficient = coeff;
		get_layer(1)->contraction_coefficient = coeff;
	}

	virtual ~CAutoencoder() {}

	virtual const char* get_name() const { return "Autoencoder"; }

protected:
	/** Computes the error between the output layer's activations and the given
	 * target activations.
	 *
	 * @param targets desired values for the network's output, matrix of size
	 * num_neurons_output_layer*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> targets);

private:
	void init();

	/** Returns the section of vector v that belongs to layer i */
	template<class T>
	SGVector<T> get_section(SGVector<T> v, int32_t i);

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

protected:
	/** For contractive autoencoders [Rifai, 2011], a term:
	 * \f[ \frac{\lambda}{N} \sum_{k=0}^{N-1} \left \| J(x_k) \right \|^2_F \f]
	 * is added to the error, where \f$ \left \| J(x_k)) \right \|^2_F \f$ is the
	 * Frobenius norm of the Jacobian of the activations of the hidden layer
	 * with respect to its inputs, \f$ N \f$ is the batch size, and
	 * \f$ \lambda \f$ is the contraction coefficient.
	 *
	 * Default value is 0.0.
	 */
	float64_t m_contraction_coefficient;
};
}
#endif

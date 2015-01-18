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

#ifndef __NEURALLAYER_H__
#define __NEURALLAYER_H__

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{

/** For autoencoders, specifies the position of the layer in the autoencoder, 
 * i.e an encoding layer or a decoding layer 
 */
enum ENLAutoencoderPosition
{
	/** The layer is not a part of an autoencoder */
	NLAP_NONE=0,
	/** The layer is an encoding layer */
	NLAP_ENCODING=1,
	/** The layer is a decoding layer */
	NLAP_DECODING=2
};	

template <class T> class SGVector;	

/** @brief Base class for neural network layers
 * 
 * A Neural layer represents an group of neurons and is the basic building block
 * for neural networks.
 * 
 * This class is to be inherited from to implement layers with different neuron
 * types (i.e linear, softmax, convolutional, etc..)
 *
 * Any arbitrary layer type can be derived from this class provided that the 
 * following functions have been defined in a mathematically plausible manner:
 * - initialize_parameters()
 * - compute_activations()
 * - compute_gradients()
 * - compute_error() [only if the layer can be used as an output layer]
 * - enforce_max_norm()
 * 
 * The memory for the layer's parameters (weights and biases) is not allocated
 * by the layer itself, instead it is allocated by the network that the layer
 * belongs to, and passed to the layer when it needs to use it.
 * 
 * This class stores buffers for use during forward and backpropagation, this is
 * to avoid unnecessary memory allocation during computations. The buffers are:
 * m_activations: size m_num_neurons*m_batch_size
 * m_activation_gradients: size m_num_neurons*m_batch_size
 * m_local_gradients: size m_num_neurons*m_batch_size
 */
class CNeuralLayer : public CSGObject
{
public:
	/** default constructor */
	CNeuralLayer();
	
	/** Constuctor
	 * 
	 * @param num_neurons Number of neurons in this layer
	 */
	CNeuralLayer(int32_t num_neurons);
	
	virtual ~CNeuralLayer();
	
	/** Initializes the layer
	 * 
	 * @param layers Array of layers that form the network that this layer is 
	 * being used with
	 * 
	 * @param input_indices  Indices of the layers that are connected to this 
	 * layer as input
	 */
	virtual void initialize(CDynamicObjectArray* layers, 
			SGVector<int32_t> input_indices);
	
	/** Sets the batch_size and allocates memory for m_activations and
	 * m_input_gradients accordingly. Must be called before forward or backward
	 * propagation is performed
	 * 
	 * @param batch_size number of training/test cases the network is 
	 * currently working with
	 */
	virtual void set_batch_size(int32_t batch_size);
	
	/** returns true if the layer is an input layer. Input layers are the root 
	 * layers of a network, that is, they don't receive signals from other
	 * layers, they receive signals from the inputs features to the network.
	 * 
	 * Local and activation gradients are not computed for input layers
	 */
	virtual bool is_input() { return false; }
	
	/** Initializes the layer's parameters. The layer should fill the given 
	 * arrays with the initial value for its parameters
	 *
	 * @param parameters Vector of size get_num_parameters()
	 * 
	 * @param parameter_regularizable Vector of size get_num_parameters().
	 * This controls which of the layer's parameter are 
	 * subject to regularization, i.e to turn off regularization for parameter 
	 * i, set parameter_regularizable[i] = false. This is usally used to turn 
	 * off regularization for bias parameters.
	 * 
	 * @param sigma standard deviation of the gaussian used to random the
	 * parameters
	 */
	virtual void initialize_parameters(SGVector<float64_t> parameters,
			SGVector<bool> parameter_regularizable,
			float64_t sigma) { }
	
	/** Computes the activations of the neurons in this layer, results should 
	 * be stored in m_activations. To be used only with input layers
	 * 
	 * @param inputs activations of the neurons in the 
	 * previous layer, matrix of size previous_layer_num_neurons * batch_size
	 */
	virtual void compute_activations(SGMatrix<float64_t> inputs) { }
	
	/** Computes the activations of the neurons in this layer, results should 
	 * be stored in m_activations. To be used only with non-input layers
	 * 
	 * @param parameters Vector of size get_num_parameters(), contains the 
	 * parameters of the layer
	 * 
	 * @param layers Array of layers that form the network that this layer is 
	 * being used with
	 */
	virtual void compute_activations(SGVector<float64_t> parameters,
			CDynamicObjectArray* layers) { }
	
	/** Computes the gradients that are relevent to this layer:
	 *- The gradients of the error with respect to the layer's parameters
	 * -The gradients of the error with respect to the layer's inputs
	 * 
	 * Input gradients for layer i that connects into this layer as input are 
	 * added to m_layers.element(i).get_activation_gradients()
	 * 
	 * Deriving classes should make sure to account for 
	 * [dropout](http://arxiv.org/abs/1207.0580) [Hinton, 2012] during gradient 
	 * computations
	 * 
	 * Deriving classes should also account for contraction_coefficient if they 
	 * can be used in as a hidden layer in a contractive autoencoder.
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the 
	 * parameters of the layer
	 * 
	 * @param targets a matrix of size num_neurons*batch_size. If the layer is 
	 * being used as an output layer, targets is the desired values for the 
	 * layer's activations, otherwise it's an empty matrix
	 * 
	 * @param layers Array of layers that form the network that this layer is 
	 * being used with
	 * 
	 * @param parameter_gradients Vector of size get_num_parameters(). To be 
	 * filled with gradients of the error with respect to each parameter of the 
	 * layer
	 */
	virtual void compute_gradients(SGVector<float64_t> parameters, 
			SGMatrix<float64_t> targets,
			CDynamicObjectArray* layers,
			SGVector<float64_t> parameter_gradients) { }
	
	/** Computes the error between the layer's current activations and the given
	 * target activations. Should only be used with output layers
	 *
	 * @param targets desired values for the layer's activations, matrix of size
	 * num_neurons*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> targets) { return 0; }
	
	/** Constrains the weights of each neuron in the layer to have an L2 norm of
	 * at most max_norm
	 * 
	 * @param parameters pointer to the layer's parameters, array of size 
	 * get_num_parameters()
	 * 
	 * @param max_norm maximum allowable norm for a neuron's weights
	 */
	virtual void enforce_max_norm(SGVector<float64_t> parameters, 
			float64_t max_norm) { }
	
	/** Applies [dropout](http://arxiv.org/abs/1207.0580) [Hinton, 2012] to the 
	 * activations of the layer 
	 * 
	 * If is_training is true, fills m_dropout_mask with random values 
	 * (according to dropout_prop) and multiplies it into the activations, 
	 * otherwise, multiplies the activations by (1-dropout_prop) to compensate 
	 * for using dropout during training
	 */
	virtual void dropout_activations();
	
	/** Computes 
	 * \f[ \frac{\lambda}{N} \sum_{k=0}^{N-1} \left \| J(x_k) \right \|^2_F \f]
	 * where \f$ \left \| J(x_k)) \right \|^2_F \f$ is the Frobenius norm of 
	 * the Jacobian of the activations of the hidden layer with respect to its 
	 * inputs, \f$ N \f$ is the batch size, and \f$ \lambda \f$ is the 
	 * contraction coefficient.
	 * 
	 * Should be implemented by layers that support being used as a hidden 
	 * layer in a contractive autoencoder.
	 * 
	 * @param parameters Vector of size get_num_parameters(), contains the 
	 * parameters of the layer
	 */
	virtual float64_t compute_contraction_term(SGVector<float64_t> parameters) 
	{ 
		return 0.0; 
	}
	
	/** Gets the number of neurons in the layer
	 * 
	 * @return number of neurons in the layer
	 */
	virtual int32_t get_num_neurons() { return m_num_neurons; }
	
	/** Returns the width assuming that the layer's activations are interpreted as 
	 * images (i.e for convolutional nets)
	 * 
	 * @return Width
	 */
	virtual int32_t get_width() { return m_width; }
	
	/** Returns the height assuming that the layer's activations are interpreted as 
	 * images (i.e for convolutional nets)
	 * 
	 * @return Height
	 */
	virtual int32_t get_height() { return m_height; }
	
	/** Gets the number of parameters used in this layer
	 * 
	 * @return number of parameters used in this layer
	 */
	virtual int32_t get_num_parameters() { return m_num_parameters; }
	
	/** Gets the layer's activations, a matrix of size num_neurons * batch_size
	 * 
	 * @return layer's activations
	 */
	virtual SGMatrix<float64_t> get_activations() { return m_activations; }
	
	/** Gets the layer's activation gradients, a matrix of size
	 * num_neurons * batch_size
	 * 
	 * @return layer's activation gradients
	 */
	virtual SGMatrix<float64_t> get_activation_gradients() 
	{
		return m_activation_gradients;
	}
	
	/** Gets the layer's local gradients, a matrix of size
	 * num_neurons * batch_size
	 * 
	 * @return layer's local gradients
	 */
	virtual SGMatrix<float64_t> get_local_gradients() 
	{
		return m_local_gradients;
	}
	
	virtual const char* get_name() const { return "NeuralLayer"; }

private:
	void init();

public:
	/** Should be true if the layer is currently used during training 
	 * initial value is false
	 */
	bool is_training;
	
	/** probabilty of dropping out a neuron in the layer */
	float64_t dropout_prop;
	
	/** For hidden layers in a contractive autoencoders [Rifai, 2011] a term:
	 * \f[ \frac{\lambda}{N} \sum_{k=0}^{N-1} \left \| J(x_k) \right \|^2_F \f] 
	 * is added to the error, where \f$ \left \| J(x_k)) \right \|^2_F \f$ is the 
	 * Frobenius norm of the Jacobian of the activations of the hidden layer 
	 * with respect to its inputs, \f$ N \f$ is the batch size, and 
	 * \f$ \lambda \f$ is the contraction coefficient. 
	 * 
	 * Default value is 0.0.
	 */ 
	float64_t contraction_coefficient;
	
	/** For autoencoders, specifies the position of the layer in the autoencoder, 
	 * i.e an encoding layer or a decoding layer. Default value is NLAP_NONE
	 */
	ENLAutoencoderPosition autoencoder_position;
	
protected:
	/** Number of neurons in this layer */
	int32_t m_num_neurons;
	
	/** Width of the image (if the layer's activations are to be interpreted as 
	 * images. Default value is m_num_neurons
	 */
	int32_t m_width;
	
	/** Width of the image (if the layer's activations are to be interpreted as 
	 * images. Default value is 1
	 */
	int32_t m_height;
	
	/** Number of neurons in this layer */
	int32_t m_num_parameters;
	
	/** Indices of the layers that are connected to this layer as input */
	SGVector<int32_t> m_input_indices;
	
	/** Number of neurons in the layers that are connected to this layer as 
	 * input 
	 */
	SGVector<int32_t> m_input_sizes;
	
	/** number of training/test cases the network is currently working with */
	int32_t m_batch_size;
	
	/** activations of the neurons in this layer
	 * size num_neurons * batch_size
	 */
	SGMatrix<float64_t> m_activations;
	
	/** gradients of the error with respect to the layer's inputs
	 * size previous_layer_num_neurons * batch_size
	 */
	SGMatrix<float64_t> m_activation_gradients;
	
	/** gradients of the error with respect to the layer's pre-activations, 
	 * this is usually used as a buffer when computing the input gradients
	 * size num_neurons * batch_size
	 */
	SGMatrix<float64_t> m_local_gradients;
	
	/** binary mask that determines whether a neuron will be kept or dropped out
	 * during the current iteration of training 
	 * size num_neurons * batch_size
	 */
	SGMatrix<bool> m_dropout_mask;
};

}
#endif

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Khaled Nasr
 */

#ifndef __NEURALLAYER_H__
#define __NEURALLAYER_H__

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
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
 * - get_num_parameters()
 * - compute_activations()
 * - compute_gradients()
 * - computer_error() [only if the layer can be used as an output layer]
 * 
 * The memory for the layer's parameters (weights and biases) is not allocated
 * by the layer itself, instead it is allocated by the network that the layer
 * belongs to, and passed to the layer when it needs to use it.
 * 
 * This class stores buffers for use during forward and backpropagation, this is
 * to avoid unnecessary memory allocation during computations. The buffers are:
 * m_activations: size m_num_neurons*m_batch_size
 * m_input_gradients: size m_num_neurons*m_previous_layer_num_neurons
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
	 * @param previous_layer_num_neurons number of neurons in the previous layer
	 */
	virtual void initialize(int32_t previous_layer_num_neurons);
	
	/** Sets the batch_size and allocates memory for m_activations and
	 * m_input_gradients accordingly. Must be called before forward or backward
	 * propagation is performed
	 * 
	 * @param batch_size number of training/test cases the network is 
	 * currently working with
	 */
	virtual void set_batch_size(int32_t batch_size);
	
	/** Gets the number of parameters (weights and biases) needed for this 
	 * layer
	 * 
	 * @return number of parameters (weights and biases) needed for this layer
	 */
	virtual int32_t get_num_parameters() = 0;
	
	/** Initializes the layer's parameters. The layer should fill the given 
	 * arrays with the initial value for its parameters
	 *
	 * @param parameters preallocated array of size get_num_parameters()
	 * 
	 * @param parameter_regularizable preallocated array of size 
	 * get_num_parameters(). This controls which of the layer's parameter are
	 * subject to regularization, i.e to turn off regularization for parameter 
	 * i, set parameter_regularizable[i] = false. This is usally used to turn 
	 * off regularization for bias parameters.
	 * 
	 * @param sigma standard deviation of the gaussian used to random the
	 * parameters
	 */
	virtual void initialize_parameters(float64_t* parameters,
			bool* parameter_regularizable,
			float64_t sigma = 0.01f) = 0;
	
	/** Computes the activations of the neurons in this layer, results should 
	 * be stored in m_activations
	 * 
	 * @param parameters pointer to the layer's parameters, array of size 
	 * get_num_parameters() 
	 * 
	 * @param previous_layer_activations activations of the neurons in the 
	 * previous layer, matrix of size previous_layer_num_neurons * batch_size
	 */
	virtual void compute_activations(float64_t* parameters,
			float64_t* previous_layer_activations) = 0;
	
	/** Computes the gradients that are relevent to this layer:
	 *- The gradients of the error with respect to the layer's parameters
	 * -The gradients of the error with respect to the layer's inputs
	 * 
	 * The input gradients are stored in m_input_gradients
	 *
	 * @param parameters pointer to the layer's parameters, array of size 
	 * get_num_parameters() 
	 * 
	 * @param is_output specifies if the layer is used as an output layer or a
	 * hidden layer
	 * 
	 * @param p a matrix of size num_neurons*batch_size. If is_output is true,p 
	 * is the desired values for the layer's activations, else it is the
	 * gradients of the error with respect to this layer's activations (the 
	 * input gradients of the next layer).
	 *
	 * @param previous_layer_activations activations of the neurons in the
	 * previous layer, matrix of size previous_layer_num_neurons * batch_size
	 * 
	 * @param parameter_gradients preallocated array of size
	 * get_num_parameters(), to be filled with the parameter gradients of this 
	 * layer
	 */
	virtual void compute_gradients(float64_t* parameters, 
			bool is_output,
			float64_t* p,
			float64_t* previous_layer_activations,
			float64_t* parameter_gradients) = 0;
	
	/** Computes the error between the layer's current activations and the given
	 * target activations. Should only be used with output layers
	 *
	 * @param targets desired values for the layer's activations, matrix of size
	 * num_neurons*batch_size
	 */
	virtual float64_t computer_error(float64_t* targets) = 0;
	
	/** Gets the number of neurons in the layer
	 * 
	 * @return number of neurons in the layer
	 */
	virtual int32_t get_num_neurons() {return m_num_neurons;}
	
	/** Gets the layer's activations, a matrix of size num_neurons * batch_size
	 * 
	 * @return layer's activations
	 */
	virtual float64_t* get_activations() {return m_activations;}
	
	/** Gets the layer's input gradients, a matrix of size
	 * previous_layer_num_neurons * batch_size
	 * 
	 * @return layer's input gradients
	 */
	virtual float64_t* get_input_gradients() {return m_input_gradients;}
	
	virtual const char* get_name() const { return "NeuralLayer"; }

private:
	void init();
	
	void shallow_copy(const CNeuralLayer &orig);
	
protected:
	/** Number of neurons in this layer */
	int32_t m_num_neurons;
	
	/** Number of neurons in the previous layer */
	int32_t m_previous_layer_num_neurons;
	
	/** number of training/test cases the network is currently working with */
	int32_t m_batch_size;
	
	/** activations of the neurons in this layer
	 * length num_neurons * batch_size
	 */
	SGVector<float64_t> m_activations;
	
	/** gradients of the error with respect to the layer's inputs
	 * length previous_layer_num_neurons * batch_size
	 */
	SGVector<float64_t> m_input_gradients;
	
	/** gradients of the error with respect to the layer's activations, this is
	 * usually used as a buffer when computing the input gradients
	 * length num_neurons * batch_size
	 */
	SGVector<float64_t> m_local_gradients;
};
	
}
#endif
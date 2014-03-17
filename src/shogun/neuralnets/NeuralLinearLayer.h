/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Khaled Nasr
 */

#ifndef __NEURALLINEARLAYER_H__
#define __NEURALLINEARLAYER_H__

#include <shogun/lib/common.h>
#include <shogun/neuralnets/NeuralLayer.h>

namespace shogun
{
/** @brief Neural layer with linear neurons, with an identity activation 
 * function. can be used as a hidden layer or an output layer
 * 
 * When used as an output layer, a squared error measure is used
 */
class CNeuralLinearLayer : public CNeuralLayer
{
public:
	/** default constructor */
	CNeuralLinearLayer();
	
	/** Constuctor
	 * 
	 * @param num_neurons Number of neurons in this layer
	 */
	CNeuralLinearLayer(int32_t num_neurons);
	
	virtual ~CNeuralLinearLayer() {}
	
	/** Gets the number of parameters (weights and biases) needed for this 
	 * layer
	 * 
	 * @return number of parameters (weights and biases) needed for this layer
	 */
	virtual int32_t get_num_parameters();
	
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
			float64_t sigma = 0.01f);
	
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
			float64_t* previous_layer_activations);
	
	/** Computes the gradients that are relevent to this layer:
	 *		- The gradients of the error with respect to the layer's parameters
	 * 		- The gradients of the error with respect to the layer's inputs
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
			float64_t* parameter_gradients);
	
	/** Computes the error between the layer's current activations and the given
	 * target activations. Should only be used with output layers
	 * 
	 * @param targets desired values for the layer's activations, matrix of size
	 * num_neurons*batch_size
	 */
	virtual float64_t computer_error(float64_t* targets);
	
	/** Computes the gradients of the error with respect to this layer's
	 * activations. Results are stored in m_local_gradients. 
	 * 
	 * This is used by compute_gradients() and can be overriden to implement 
	 * layers with different activation functions
	 *
	 * @param is_output specifies if the layer is used as an output layer or a
	 * hidden layer
	 * 
	 * @param p a matrix of size num_neurons*batch_size. If is_output is true,p 
	 * is the desired values for the layer's activations, else it is the
	 * gradients of the error with respect to this layer's activations (the 
	 * input gradients of the next layer).
	 */
	virtual void compute_local_gradients(bool is_output, float64_t* p);
	
	virtual const char* get_name() const { return "NeuralLinearLayer"; }
};
	
}
#endif

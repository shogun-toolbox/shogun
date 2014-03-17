/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Khaled Nasr
 */

#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/neuralnets/NeuralLayer.h>

namespace shogun
{
/** @brief A generic multi-layer neural network
 *
 * NeuralNetwork is constructed using an array of NeuralLayer objects. The
 * NeuralLayer class defines the interface necessary for forward and 
 * backpropagation.
 * 
 * The network stores the parameters (and parameter gradients) of all the 
 * layers in a single array. This makes it easy to train a network of any
 * combination of arbitrary layer types using any optimization method (gradient
 * descent, L-BFGS, ..)
 * 
 * The network can use L2 regularization during gradient computation, this is
 * enabled by calling set_L2_regularization()
 * 
 * All the matrices the network (and related classes) deal with are in 
 * column-major format
 * 
 * When implemnting new layer types, the function check_gradients() can be used
 * to make sure the gradient computations are correct.
 */
class CNeuralNetwork : public CSGObject
{
public:
	/** default constuctor */
	CNeuralNetwork();
	
	/** Initializes the network
	 * 
	 * @param num_inputs number of inputs the network takes
	 * 
	 * @param layers An array of NeuralLayer objects specifying the hidden 
	 * and output layers in the network.
	 */
	virtual void initialize(int32_t num_inputs, CDynamicObjectArray* layers);
	
	virtual ~CNeuralNetwork();
	
	/** Sets the L2 regularization coefficient used during gradient computation
	 * 
	 * @param L2_coeff L2 regularization coefficient, if 0.0, L2 regularization 
	 * is disabled
	 */
	virtual void set_L2_regularization(float64_t L2_coeff)
	{
		m_L2_coeff = L2_coeff;
	}
	
	/** Computes the output of the network
	 * 
	 * @param inputs inputs matrix, size: num_inputs*num_cases,
	 * a column for each test case
	 * 
	 * @return pointer to a newly allocated outputs matrix
	 * size: num_neurons_output_layer*num_cases, a column for each test case
	 */
	virtual CDenseFeatures<float64_t>* apply(CDenseFeatures<float64_t>* inputs);
	
	/** Trains the network using the mini-batch gradient descent algorithm
	 * 
	 * @param inputs training examples, matrix of size num_inputs*num_cases
	 * @param targets desired values for the network's output, matrix of size
	 * num_inputs*num_cases
	 * @param learning_rate learning rate
	 * @param batch_size mini-batch size, if 0 the entire training set is used
	 * @param max_num_epochs maximum number of iterations over the training set
	 * @param momentum momentum multiplier
	 */
	virtual void train_gradient_descent(CDenseFeatures<float64_t>* inputs,
			CDenseFeatures<float64_t>* targets,
			int32_t max_num_epochs = 1000,
			int32_t batch_size = 0,
			float64_t learning_rate = 0.1,
			float64_t momentum = 0.9);
	
	/** Checks if the gradients computed using backpropagation are correct by 
	 * comparing them with gradients computed using numerical approximation.
	 * Used for testing purposes only.
	 * 
	 * @param epsilon constant used during gradient approximation
	 * 
	 * @param tolerance maximum difference allowed between backpropagation 
	 * gradients and numerical approximation gradients
	 * 
	 * @return true if the gradients are correct, false otherwise
	 */
	virtual bool check_gradients(float64_t epsilon=1.0e-06, 
			float64_t tolerance=1.0e-09);
	
	/** returns the totat number of parameters in the network */
	int32_t get_num_parameters() {return m_total_num_parameters;}
	
	/** returns a pointer to the network's parameter array */
	float64_t* get_parameters() {return m_params;}
	
	/** returns a pointer to the network's parameter gradients array */
	float64_t* get_parameter_gradients() {return m_param_gradients;}
	
	/** returns the number of inputs the network takes*/
	int32_t get_num_inputs() {return m_num_inputs;}
	
	/** returns the number of neurons in the output layer */
	int32_t get_num_outputs()
	{
		return get_layer(m_num_layers-1)->get_num_neurons();
	}
	
	virtual const char* get_name() const { return "NeuralNetwork";}
	
protected:
	/** Applies forward propagation, computes the activations of each layer
	 * 
	 * @param inputs inputs to the network, a matrix of size 
	 * m_input_layer_num_neurons*m_batch_size
	 */
	virtual void forward_propagate(float64_t* inputs);
	
	/** Sets the batch size (the number of train/test cases) the network is 
	 * expected to deal with. 
	 * Allocates memory for the activations, local gradients, input gradients
	 * if necessary (if the batch size is different from it's previous value)
	 * 
	 * @param batch_size number of train/test cases the network is expected to 
	 * deal with.
	 */
	virtual void set_batch_size(int32_t batch_size);
	
	/** Applies backpropagation to compute the gradients of the error with
	 * repsect to every parameter in the network. Results are stored in 
	 * m_param_gradients and can be accessed by calling 
	 * get_parameter_gradients()
	 *
	 * @param inputs inputs to the network, a matrix of size 
	 * m_input_layer_num_neurons*m_batch_size
	 * 
	 * @param targets desired values for the output layer's activations. matrix 
	 * of size m_layers[m_num_layers-1].get_num_neurons()*m_batch_size
	 */
	virtual void compute_gradients(float64_t* inputs, float64_t* targets);
	
	/** Computes the error between the output layer's activations and the given
	 * target activations.
	 *
	 * Regularization error is ignored.
	 * 
	 * @param targets desired values for the network's output, matrix of size
	 * num_neurons_output_layer*batch_size
	 * 
	 * @param inputs if NULL, the error is computed between the current 
	 * activations and the given targets, no forward propagation is performed.
	 * otherwise [if inputs is a matrix of size 
	 * input_layer_num_neurons*batch_size], forward propagation is performed to
	 * update the activations before the error is computed.
	 */
	virtual float64_t compute_error(float64_t* targets, 
			float64_t* inputs=NULL);
	
private:
	/** returns a pointer to layer i in the network */
	CNeuralLayer* get_layer(int32_t i)
	{
		return (CNeuralLayer*)m_layers->element(i);
	}
	
	/** returns a pointer to the portion of m_params that belongs to layer i */
	float64_t* get_layer_params(int32_t i)
	{
		return m_params.vector+m_index_offsets[i];
	}
	
	/** returns a pointer to the portion of m_param_gradients that belongs to
	 * layer i
	 */
	float64_t* get_layer_param_gradients(int32_t i)
	{
		return m_param_gradients.vector+m_index_offsets[i];
	}
	
	/** returns a pointer to the portion of m_param_regularizable that belongs
	 * to layer i
	 */
	bool* get_layer_param_regularizable(int32_t i)
	{
		return m_param_regularizable.vector+m_index_offsets[i];
	}
	
	void init();
	
protected:
	/** number of neurons in the input layer */
	int32_t m_num_inputs;
	
	/** number of layer */
	int32_t m_num_layers;
	
	/** network's layers */
	CDynamicObjectArray* m_layers;
	
	/** L2 regularization coeff, defualt value is 0.0 */
	float64_t m_L2_coeff;
	
	/** total number of parameters in the network */
	int32_t m_total_num_parameters;
	
	/** array where all the parameters of the network are stored */
	SGVector<float64_t> m_params;
	
	/** array where the gradients of the error with respect to the parameters 
	 * are stored
	 */
	SGVector<float64_t> m_param_gradients;
	
	/** Array that specifies which parameters are to be regularized. This is 
	 * used to turn off regularization for bias parameters
	 */
	SGVector<bool> m_param_regularizable;
	
	/** offsets specifying where each layer's parameters and parameter 
	 * gradients are stored, i.e layer i's parameters are stored at 
	 * m_params + m_index_offsets[i]
	 */
	SGVector<int32_t> m_index_offsets;
	
	/** number of train/test cases the network is expected to deal with.
	 * defaul value is 1
	 */
	int32_t m_batch_size;
};
	
}
#endif

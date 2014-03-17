/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Khaled Nasr
 */

#ifndef __NEURALLOGISTICLAYER_H__
#define __NEURALLOGISTICLAYER_H__

#include <shogun/neuralnets/NeuralLinearLayer.h>

namespace shogun
{
/** @brief Neural layer with linear neurons, with a logistic activation 
 * function. can be used as a hidden layer or an output layer
 * 
 * When used as an output layer, a squared error measure is used
 */
class CNeuralLogisticLayer : public CNeuralLinearLayer
{
public:
	/** default constructor */
	CNeuralLogisticLayer();
	
	/** Constuctor
	 * 
	 * @param num_neurons Number of neurons in this layer
	 */
	CNeuralLogisticLayer(int32_t num_neurons);
	
	virtual ~CNeuralLogisticLayer() {}
	
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
	 * 
	 * @return if is_output is true returns the error, else retruns 0
	 */
	virtual void compute_local_gradients(bool is_output, float64_t* p);
	
	virtual const char* get_name() const { return "NeuralLogisticLayer"; }
};
	
}
#endif

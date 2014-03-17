/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Khaled Nasr
 */

#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CNeuralLogisticLayer::CNeuralLogisticLayer() : CNeuralLinearLayer()
{
}

CNeuralLogisticLayer::CNeuralLogisticLayer(int32_t num_neurons): 
CNeuralLinearLayer(num_neurons)
{
}

void CNeuralLogisticLayer::compute_activations(float64_t* parameters,
		float64_t* previous_layer_activations)
{
	CNeuralLinearLayer::compute_activations(parameters, 
		previous_layer_activations);
	
	// apply logistic activation function
	int32_t length = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<length; i++)
		m_activations[i] = 1.0/(1.0+CMath::exp(-1.0*m_activations[i]));
}

void CNeuralLogisticLayer::compute_local_gradients(bool is_output, 
		float64_t* p)
{
	CNeuralLinearLayer::compute_local_gradients(is_output,p);
	
	// multiply by the derivative of the logistic function
	int32_t length = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<length; i++)
		m_local_gradients[i] *= m_activations[i] * (1.0-m_activations[i]);
}

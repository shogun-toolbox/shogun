/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Khaled Nasr
 */

#include <shogun/neuralnets/NeuralLinearLayer.h>
#include <shogun/mathematics/Random.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

using namespace shogun;

CNeuralLinearLayer::CNeuralLinearLayer() : CNeuralLayer()
{
}

CNeuralLinearLayer::CNeuralLinearLayer(const CNeuralLinearLayer& orig)
: CNeuralLayer(orig)
{

}


CNeuralLinearLayer::CNeuralLinearLayer(int32_t num_neurons): 
CNeuralLayer(num_neurons)
{	
}

int32_t CNeuralLinearLayer::get_num_parameters()
{
	// A weight matrix of size num_neurons*previous_layer_num_neurons and vector
	// of biases of size num_neurons
	return m_num_neurons*m_previous_layer_num_neurons + m_num_neurons;
}

void CNeuralLinearLayer::initialize_parameters(float64_t* parameters, 
		bool* parameter_regularizable, 
		float64_t sigma)
{
	CRandom random_generator(CRandom::generate_seed());
	
	// random the parameters
	int32_t num_parameters = get_num_parameters();
	for (int32_t i=0; i<num_parameters; i++)
		parameters[i] = random_generator.normal_distrib(0.0, sigma);
	
	// turn regularization off for the biases, on for the weights
	int32_t num_weights = m_num_neurons*m_previous_layer_num_neurons;
	for (int32_t i=0; i<num_parameters; i++)
		parameter_regularizable[i] = (i<num_weights);
}


void CNeuralLinearLayer::compute_activations(float64_t* parameters, 
		float64_t* previous_layer_activations)
{
	float64_t* weights = parameters;
	float64_t* biases = parameters + m_num_neurons*m_previous_layer_num_neurons;
	
#ifdef HAVE_EIGEN3
	Eigen::Map<Eigen::MatrixXd> W(weights, m_num_neurons, 
			m_previous_layer_num_neurons);
	Eigen::Map<Eigen::MatrixXd> X(previous_layer_activations, 
			m_previous_layer_num_neurons, m_batch_size);
	Eigen::Map<Eigen::MatrixXd>  A(m_activations, m_num_neurons, m_batch_size);
	Eigen::Map<Eigen::VectorXd>  B(biases, m_num_neurons);
	
	A = W*X;
	A.colwise() += B;
#endif
}

void CNeuralLinearLayer::compute_gradients(float64_t* parameters,
		bool is_output, float64_t* p,
		float64_t* previous_layer_activations,
		float64_t* parameter_gradients)
{
	float64_t* weights = parameters;
	
	compute_local_gradients(is_output, p);
	
#ifdef HAVE_EIGEN3
	Eigen::Map<Eigen::MatrixXd> X(previous_layer_activations, 
			m_previous_layer_num_neurons, m_batch_size);
	Eigen::Map<Eigen::MatrixXd>  W(weights, m_num_neurons, 
			m_previous_layer_num_neurons);
	Eigen::Map<Eigen::MatrixXd> LG(m_local_gradients, m_num_neurons, 
			m_batch_size);
	Eigen::Map<Eigen::MatrixXd> WG(parameter_gradients, 
			m_num_neurons, m_previous_layer_num_neurons);
	Eigen::Map<Eigen::VectorXd> BG(parameter_gradients + 
				m_num_neurons*m_previous_layer_num_neurons, m_num_neurons);
	Eigen::Map<Eigen::MatrixXd>  IG(m_input_gradients, 
			   m_previous_layer_num_neurons, m_batch_size);
	
	// compute parameter gradients
	WG = (LG*X.transpose())/m_batch_size;
	BG = LG.rowwise().sum()/m_batch_size;
	
	// compute input gradients
	IG = W.transpose()*LG;
#endif	
}

void CNeuralLinearLayer::compute_local_gradients(bool is_output, float64_t* p)
{	
	if (is_output)
	{
		// sqaured error measure
		// local_gradients = activations-targets
		int32_t length = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<length; i++)
			m_local_gradients[i] = m_activations[i]-p[i];
		
	}
	else
	{
		int32_t length = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<length; i++)
			m_local_gradients[i] = p[i];
	}
	
}

float64_t CNeuralLinearLayer::computer_error(float64_t* targets)
{
	// error = 0.5*(sum(targets-activations)^2)/batch_size
	float64_t sum = 0;
	int32_t length = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<length; i++)
		sum += (targets[i]-m_activations[i])*(targets[i]-m_activations[i]);
	sum *= (0.5/m_batch_size);
	return sum;
}

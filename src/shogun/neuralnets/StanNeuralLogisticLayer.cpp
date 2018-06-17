/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Khaled Nasr
 */

 #include <shogun/neuralnets/StanNeuralLogisticLayer.h>
 #include <shogun/mathematics/Math.h>
 #include <shogun/lib/SGVector.h>

 using namespace shogun;

 StanNeuralLogisticLayer::StanNeuralLogisticLayer() : StanNeuralLinearLayer()
 {
 }

 StanNeuralLogisticLayer::StanNeuralLogisticLayer(int32_t num_neurons):
 StanNeuralLinearLayer(num_neurons)
 {
 }

 void StanNeuralLogisticLayer::compute_activations(StanVector& parameters,
 		CDynamicObjectArray* layers)
 {
 	StanNeuralLinearLayer::compute_activations(parameters, layers);

  for(index_t i=0; i< m_num_neurons; ++i)
  {
    for(index_t j=0; j<m_batch_size; ++j)
    {
      m_stan_activations(i,j) = 1.0 / (1.0 + stan::math::exp(-1.0 * m_stan_activations(i,j) ) );
    }
  }
 }

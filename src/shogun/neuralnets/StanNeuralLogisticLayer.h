/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Khaled Nasr
 */

#ifndef __STANNEURALLOGISTICLAYER_H__
#define __STANNEURALLOGISTICLAYER_H__

#include <shogun/neuralnets/StanNeuralLinearLayer.h>

namespace shogun
{
/** @brief Neural layer with linear neurons, with a [logistic activation
 * function](http://en.wikipedia.org/wiki/Logistic_function). can be used as a
 * hidden layer or an output layer.
 *
 * When used as an output layer, a
 * [squared error measure](http://en.wikipedia.org/wiki/Mean_squared_error) is
 * used
 */
class StanNeuralLogisticLayer : public StanNeuralLinearLayer
{
public:
	/** default constructor */
	StanNeuralLogisticLayer();

	/** Constructor
	 *
	 * @param num_neurons Number of neurons in this layer
	 */
	StanNeuralLogisticLayer(int32_t num_neurons);

	virtual ~StanNeuralLogisticLayer() {}

	/** Computes the activations of the neurons in this layer, results should
	 * be stored in m_activations. To be used only with non-input layers
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 */
	virtual void compute_activations(StanVector& parameters,
			CDynamicObjectArray* layers);


	virtual const char* get_name() const { return "StanNeuralLogisticLayer"; }
};

}
#endif

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Manjunath Bhat
 */
#ifndef __NEURALMISHLAYER_H__
#define __NEURALMISHLAYER_H__

#include <shogun/neuralnets/NeuralLinearLayer.h>

namespace shogun
{
/** @brief Neural layer with [Mish activated neurons]
 * (https://arxiv.org/abs/1908.08681).
 *
 * Activations are computed according to x*tanh(log(1 + e^x))
 *
 * When used as an output layer, a squared error measure is used
 */
class NeuralMishLayer : public NeuralLinearLayer
{
public:
	/** default constructor */
	NeuralMishLayer();

	/** Constuctor
	 *
	 * @param num_neurons Number of neurons in this layer
	 */
	NeuralMishLayer(int32_t num_neurons);

	virtual ~NeuralMishLayer() {}

	/** Computes the activations of the neurons in this layer, results should
	 * be stored in m_activations. To be used only with non-input layers
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 */
	virtual void compute_activations(
		SGVector<float64_t> parameters,
		const std::vector<std::shared_ptr<NeuralLayer>>& layers);

	virtual const char* get_name() const { return "NeuralMishLayer"; }
};

}
#endif //__NEURALMISHLAYER_H__

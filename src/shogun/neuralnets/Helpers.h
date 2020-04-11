/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef __NEURALNETSHELPERS_H__
#define __NEURALNETSHELPERS_H__

#include <shogun/machine/Machine.h>

namespace shogun {
	std::shared_ptr<DenseFeatures<float64_t>> reconstruct(const std::shared_ptr<Machine>& net, 
		const std::shared_ptr<Features>& data);

	std::shared_ptr<Machine> convert_to_neural_network(const std::shared_ptr<Machine>& net, 
		std::shared_ptr<NeuralLayer> output_layer, float64_t sigma);
}

#endif
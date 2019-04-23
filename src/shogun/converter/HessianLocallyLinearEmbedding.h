/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer
 */

#ifndef HESSIANLOCALLYLINEAREMBEDDING_H_
#define HESSIANLOCALLYLINEAREMBEDDING_H_
#include <shogun/lib/config.h>
#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Features;
class Distance;

/** @brief class HessianLocallyLinearEmbedding used to preprocess
 * data using Hessian Locally Linear Embedding algorithm as described in
 *
 * Donoho, D., & Grimes, C. (2003).
 * Hessian eigenmaps: new tools for nonlinear dimensionality reduction.
 * Proceedings of National Academy of Science (Vol. 100, pp. 5591-5596).
 *
 * Be sure k value is set with at least
 * 1+[target dim]+1/2 [target_dim]*[1 + target dim], e.g.
 * greater than 6 for target dimensionality of 2.
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','hlle',k);
 *
 */
class HessianLocallyLinearEmbedding: public LocallyLinearEmbedding
{
public:

	/** constructor */
	HessianLocallyLinearEmbedding();

	/** destructor */
	virtual ~HessianLocallyLinearEmbedding();

	/** get name */
	virtual const char* get_name() const;

	/** transform */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);
};
}

#endif /* HESSIANLOCALLYLINEAREMBEDDING_H_ */

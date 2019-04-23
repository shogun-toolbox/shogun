/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer
 */

#ifndef NEIGHBORHOODPRESERVINGEMBEDDING_H_
#define NEIGHBORHOODPRESERVINGEMBEDDING_H_
#include <shogun/lib/config.h>
#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Features;
class Distance;

/** @brief NeighborhoodPreservingEmbedding converter used to
 * construct embeddings as described in:
 *
 * He, X., Cai, D., Yan, S., & Zhang, H.-J. (2005).
 * Neighborhood preserving embedding.
 * Tenth IEEE International Conference on Computer Vision ICCV05 Volume 1, 2, 1208-1213. Ieee.
 * Retrieved from http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=1544858
 *
 * This method is hardly applicable to very high-dimensional data due to
 * necessity to solve eigenproblem involving matrix of size (dim x dim).
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','npe',k);
 *
 */
class NeighborhoodPreservingEmbedding: public LocallyLinearEmbedding
{
public:

	/** constructor */
	NeighborhoodPreservingEmbedding();

	/** destructor */
	virtual ~NeighborhoodPreservingEmbedding();

	/** get name */
	virtual const char* get_name() const;

	/** transform */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);
};
}

#endif /* NEIGHBORHOODPRESERVINGEMBEDDING_H_ */

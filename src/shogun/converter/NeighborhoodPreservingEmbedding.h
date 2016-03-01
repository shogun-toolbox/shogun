/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Sergey Lisitsyn
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

class CFeatures;
class CDistance;

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
class CNeighborhoodPreservingEmbedding: public CLocallyLinearEmbedding
{
public:

	/** constructor */
	CNeighborhoodPreservingEmbedding();

	/** destructor */
	virtual ~CNeighborhoodPreservingEmbedding();

	/** get name */
	virtual const char* get_name() const;

	/** apply */
	virtual CFeatures* apply(CFeatures* features);

};
}

#endif /* NEIGHBORHOODPRESERVINGEMBEDDING_H_ */

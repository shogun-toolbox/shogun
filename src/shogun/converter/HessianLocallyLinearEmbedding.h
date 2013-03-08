/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef HESSIANLOCALLYLINEAREMBEDDING_H_
#define HESSIANLOCALLYLINEAREMBEDDING_H_
#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

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
class CHessianLocallyLinearEmbedding: public CLocallyLinearEmbedding
{
public:

	/** constructor */
	CHessianLocallyLinearEmbedding();

	/** destructor */
	virtual ~CHessianLocallyLinearEmbedding();

	/** get name */
	virtual const char* get_name() const;

	/** apply */
	virtual CFeatures* apply(CFeatures* features);
};
}

#endif /* HAVE_EIGEN3 */
#endif /* HESSIANLOCALLYLINEAREMBEDDING_H_ */

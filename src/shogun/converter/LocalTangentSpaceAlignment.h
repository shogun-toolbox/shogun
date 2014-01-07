/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LOCALTANGENTSPACEALIGNMENT_H_
#define LOCALTANGENTSPACEALIGNMENT_H_
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <converter/LocallyLinearEmbedding.h>
#include <features/Features.h>
#include <distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class LocalTangentSpaceAlignment used to embed
 * data using Local Tangent Space Alignment (LTSA)
 * algorithm as described in:
 *
 * Zhang, Z., & Zha, H. (2002). Principal Manifolds
 * and Nonlinear Dimension Reduction via Local Tangent Space Alignment.
 * Journal of Shanghai University English Edition, 8(4), 406-424. SIAM.
 * Retrieved from http://arxiv.org/abs/cs/0212008
 *
 * This algorithm is pretty stable for variations of k parameter value but
 * be sure it is set with a consistent value (at least 3-5) for reasonable
 * results.
 *
 * Uses implementation from the Tapkee library.
 *
 */
class CLocalTangentSpaceAlignment: public CLocallyLinearEmbedding
{
public:

	/** constructor */
	CLocalTangentSpaceAlignment();

	/** destructor */
	virtual ~CLocalTangentSpaceAlignment();

	/** get name */
	virtual const char* get_name() const;

	/** apply */
	virtual CFeatures* apply(CFeatures* features);

};
}

#endif /* HAVE_EIGEN3 */
#endif /* LOCALTANGENTSPACEALINGMENT_H_ */

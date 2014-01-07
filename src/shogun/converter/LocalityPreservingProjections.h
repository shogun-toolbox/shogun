/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LOCALITYPRESERVINGPROJECTIONS_H_
#define LOCALITYPRESERVINGPROJECTIONS_H_
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <converter/LaplacianEigenmaps.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class LocalityPreservingProjections used to compute
 * embeddings of data using Locality Preserving Projections method
 * as described in
 *
 * He, X., & Niyogi, P. (2003).
 * Locality Preserving Projections.
 * Matrix, 16(December), 153–160. Citeseer.
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.7147&rep=rep1&type=pdf
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','lpp',k,width);
 *
 */
class CLocalityPreservingProjections: public CLaplacianEigenmaps
{
public:

	/** constructor */
	CLocalityPreservingProjections();

	/** destructor */
	virtual ~CLocalityPreservingProjections();

	/** get name */
	virtual const char* get_name() const;

	/** apply */
	virtual CFeatures* apply(CFeatures* features);

};
}

#endif /* HAVE_EIGEN3 */
#endif /* LOCALITYPRESERVINGPROJECTIONS_H_ */

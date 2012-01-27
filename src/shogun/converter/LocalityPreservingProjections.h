/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LOCALITYPRESERVINGPROJECTIONS_H_
#define LOCALITYPRESERVINGPROJECTIONS_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/LaplacianEigenmaps.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class LocalityPreservingProjections (part of the
 * Efficient Dimensionality Reduction Toolkit) used to compute
 * embeddings of data using Locality Preserving Projections method
 * as described in
 *
 * He, X., & Niyogi, P. (2003). 
 * Locality Preserving Projections. 
 * Matrix, 16(December), 153–160. Citeseer. 
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.7147&rep=rep1&type=pdf
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

protected:

	/** construct embedding 
	 * @param features features
	 * @param W_matrix W matrix to be used
	 */
	virtual CSimpleFeatures<float64_t>* construct_embedding(CFeatures* features, 
	                                                        SGMatrix<float64_t> W_matrix);

};
}

#endif /* HAVE_LAPACK */
#endif /* LOCALITYPRESERVINGPROJECTIONS_H_ */

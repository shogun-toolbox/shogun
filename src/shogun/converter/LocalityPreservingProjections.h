/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer
 */

#ifndef LOCALITYPRESERVINGPROJECTIONS_H_
#define LOCALITYPRESERVINGPROJECTIONS_H_
#include <shogun/lib/config.h>
#include <shogun/converter/LaplacianEigenmaps.h>

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

	/** transform */
	virtual CFeatures* transform(CFeatures* features, bool inplace = true);
};
}

#endif /* LOCALITYPRESERVINGPROJECTIONS_H_ */

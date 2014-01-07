/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef ISOMAP_H_
#define ISOMAP_H_
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <converter/MultidimensionalScaling.h>
#include <io/SGIO.h>
#include <features/DenseFeatures.h>
#include <features/Features.h>
#include <distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class Isomap used to embed data using Isomap algorithm
 * as described in
 *
 * Silva, V. D., & Tenenbaum, J. B. (2003).
 * Global versus local methods in nonlinear dimensionality reduction.
 * Advances in Neural Information Processing Systems 15, 15(Figure 2), 721-728. MIT Press.
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.3407&rep=rep1&type=pdf
 *
 * It is possible to apply preprocessor to specified distance using
 * apply_to_distance.
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','isomap',k);
 *
 */
class CIsomap: public CMultidimensionalScaling
{
public:

	/* constructor */
	CIsomap();

	/* destructor */
	virtual ~CIsomap();

	/** get name */
	const char* get_name() const;

	/** setter for k parameter
	 * @param k value
	 */
	void set_k(int32_t k);

	/** getter for k parameter
	 * @return k value
	 */
	int32_t get_k() const;

	/** embed distance */
	virtual CDenseFeatures<float64_t>* embed_distance(CDistance* distance);

/// HELPERS
protected:

	/** default init */
	virtual void init();

/// FIELDS
protected:

	/** k, number of neighbors for K-Isomap */
	int32_t m_k;

};
}
#endif /* HAVE_EIGEN3 */
#endif /* ISOMAP_H_ */

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
#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief The Isomap class is used to embed data using Isomap algorithm
 * as described in:
 *
 * Silva, V. D., & Tenenbaum, J. B. (2003).
 * Global versus local methods in nonlinear dimensionality reduction.
 * Advances in Neural Information Processing Systems 15, 15(Figure 2), 721-728. MIT Press.
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.3407&rep=rep1&type=pdf
 *
 * The Isomap algorithm can be considered as a modification of the
 * classic Multidimensional Scaling algorithm. The algorithm itself
 * consists of the following steps:
 * - For each feature vector \f$x\in X\f$ find its \f$k\f$ nearest
 *   neighbours and construct the sparse neighbourhood graph.
 * - Compute the squared distances matrix \f$D\f$ such as \f$D\{i,j\} = d^2(x_i,x_j)\f$.
 * - Relax distances with shortest (so-called geodesic) distances on the sparse neighbourhood graph (e.g. with Dijkstra's algorithm).
 * - Center the matrix \f$D\f$ with subtracting row mean, column mean and adding to the grand mean. Multiply \f$D\f$ element-wise with \f$-0.5\f$.
 * - Compute embedding with the \f$t\f$ eigenvectors that correspond to the largest eigenvalues of the matrix \f$D\f$; normalize these vectors
 *   dividing each eigenvector by the square root of its corresponding eigenvalue. Form the final embedding with eigenvectors as rows and projected
 *   feature vectors as columns.

 * It is possible to apply preprocessor to specified distance using
 * apply_to_distance.
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','isomap',k);

 * cf. http://tapkee.lisitsyn.me/
 * cf. https://en.wikipedia.org/wiki/Isomap
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

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef ISOMAP_H_
#define ISOMAP_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class Isomap (part of the Efficient Dimension
 * Reduction Toolkit) used to embed data using Isomap algorithm
 * as described in
 * 
 * Silva, V. D., & Tenenbaum, J. B. (2003). 
 * Global versus local methods in nonlinear dimensionality reduction. 
 * Advances in Neural Information Processing Systems 15, 15(Figure 2), 721-728. MIT Press. 
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.3407&rep=rep1&type=pdf
 *
 * Shortest paths are being computed with Dijkstra's algorithm with heap
 * in parallel. Due to sparsity of the kNN graph Fibonacci Heap with
 * amortized O(1) Extract-Min operation time complexity is used.
 *
 * It is possible to apply preprocessor to specified distance using
 * apply_to_distance.
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

/// HELPERS
protected:

	/** default init */
	virtual void init();

	/** process distance matrix (redefined in isomap, for mds does nothing)
	 * @param distance_matrix distance matrix
	 * @return processed distance matrix
	 */
	virtual SGMatrix<float64_t> process_distance_matrix(SGMatrix<float64_t> distance_matrix);


/// FIELDS
protected:

	/** k, number of neighbors for K-Isomap */
	int32_t m_k;

/// THREADS
protected:

	/** run dijkstra thread
	 * @param p thread params
	 */
	static void* run_dijkstra_thread(void* p);

	/** approximate geodesic distance with shortest path in kNN graph
	 * @param D_matrix distance matrix (deleted on exit)
	 * @return approximate geodesic distance matrix
	 */
	SGMatrix<float64_t> isomap_distance(SGMatrix<float64_t> D_matrix);

};
}
#endif /* HAVE_LAPACK */
#endif /* ISOMAP_H_ */

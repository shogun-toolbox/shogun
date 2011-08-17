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
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/MultidimensionalScaling.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/distance/EuclidianDistance.h>

namespace shogun
{

class CFeatures;

class CDistance;

/** @brief the class Isomap used to preprocess data using K-Isomap algorithm
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
 */
class CIsomap: public CMultidimensionalScaling
{
public:

	/* constructor */
	CIsomap();

	/* destructor */
	virtual ~CIsomap();

	/** empty init
	 */
	virtual bool init(CFeatures* features) { return true; };

	/** empty cleanup
	 */
	virtual void cleanup() {};

	/** apply preprocessor to CDistance using
	 * Isomap of specified type
	 * @param distance distance
	 * @return new features with euclidean distance similar to geodesic
	 */
	virtual CSimpleFeatures<float64_t>* apply_to_distance(CDistance* distance);

	/** apply preprocessor to feature matrix using 
	 * Isomap of specified type
	 * @param features 
	 * @return new feature matrix with euclidean distance similar to geodesic
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);
	
	/** apply preprocessor to feature vector
	 * @param vector
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

	/** get name */
	virtual inline const char* get_name() const { return "Isomap"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_ISOMAP; };

	/** setter for k parameter
	 * @param k
	 */
	void inline set_k(int32_t k)
	{
		ASSERT(k>=1);
		m_k = k;
	}

	/** getter for k parameter
	 * @return k value
	 */
	int32_t inline get_k()
	{
		return m_k;
	}


protected:

	/** k, number of neighbors for K-Isomap */
	int32_t m_k;

protected:

	/** init */
	void init();

	/** run dijkstra thread
	 * p thread params
	 */
	static void* run_dijkstra_thread(void* p);

	/** approximate geodesic distance with shortest path in kNN graph
	 * @param distance given distance for shortest path computing
	 * @return custom distance with approximate geodesic distance matrix
	 */
	CCustomDistance* isomap_distance(CDistance* distance);

};
}
#endif /* HAVE_LAPACK */

#endif /* ISOMAP_H_ */

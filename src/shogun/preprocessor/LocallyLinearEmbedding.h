/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LOCALLYLINEAREMBEDDING_H_
#define LOCALLYLINEAREMBEDDING_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief the class LocallyLinearEmbedding used to preprocess
 * data using Locally Linear Embedding algorithm described in
 *
 * Saul, L. K., Ave, P., Park, F., & Roweis, S. T. (2001).
 * An Introduction to Locally Linear Embedding. Available from, 290(5500), 2323-2326.
 * Retrieved from:
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.123.7319&rep=rep1&type=pdf
 *
 * The process of finding nearest neighbors is parallel and 
 * involves Fibonacci Heap and Euclidian distance.
 *
 * Linear reconstruction step runs in parallel for objects and 
 * involves LAPACK routine DPOSV for solving a system of linear equations.
 *
 * The eigenproblem stated in the algorithm is solved with LAPACK routine 
 * DSYEVR or with ARPACK DSAUPD/DSEUPD routines if available.
 *
 * Due to computation speed, ARPACK is being used with small 
 * regularization of weight matrix and Cholesky factorization is used
 * internally for Lanzcos iterations. If the results aren't reasonable
 * LUP factorization could be used with posdef parameter set to 
 * false using set_posdef.
 *
 */
class CLocallyLinearEmbedding: public CDimensionReductionPreprocessor
{
public:

	/** constructor */
	CLocallyLinearEmbedding();

	/** destructor */
	virtual ~CLocallyLinearEmbedding();

	/** init
	 * @param features
	 */
	virtual bool init(CFeatures* features);

	/** cleanup
	 */
	virtual void cleanup();

	/** apply preprocessor to features
	 * @param features
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preprocessor to feature vector, not supported for LLE
	 * @param vector
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

	/** setter for k parameter
	 * @param k k
	 */
	void inline set_k(int32_t k)
	{
		m_k = k;
	}

	/** getter for k parameter
	 * @return k value
	 */
	int32_t inline get_k()
	{
		return m_k;
	}

	/** setter for posdef parameter
	 * @param posdef posdef value
	 */
	void inline set_posdef(bool posdef)
	{
		m_posdef = posdef;
	}

	/** getter for posdef parameter
	 * @return posdef value
	 */
	bool inline get_posdef()
	{
		return m_posdef;
	}

	/** get name */
	virtual inline const char* get_name() const { return "LocallyLinearEmbedding"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_LOCALLYLINEAREMBEDDING; };

protected:

	/** default init */
	void init();

	/** find null space of given matrix 
	 * @param matrix given matrix
	 * @param dimension dimension of null space to be computed
	 * @param force_lapack true if lapack should be used
	 * @return null-space approximation feature matrix
	 */
	SGMatrix<float64_t> find_null_space(SGMatrix<float64_t> matrix, int dimension);

	/** construct neighborhood matrix by distance
	 * @param distance_matrix distance matrix to be used
	 * @return matrix containing indexes of neighbors of i-th object
	 * in i-th column
	 */
	SGMatrix<int32_t> get_neighborhood_matrix(SGMatrix<float64_t> distance_matrix);

protected:

	/** number of neighbors */
	int32_t m_k;

	/** boolean indicating if matrix should be considered as positive-definite */
	bool m_posdef;

public:

	static const int32_t ADAPTIVE_K = -1;

protected:

	/** runs neighborhood determination thread
	 * @param p thread params
	 */
	static void* run_neighborhood_thread(void* p);

	/** runs linear reconstruction thread
	 * @param p thread params
	 */
	static void* run_linearreconstruction_thread(void* p);

	/** runs sparse matrix-matrix multiplication thread
	 * @param p thread params
	 */
	static void* run_sparsedot_thread(void* p);


};
}

#endif /* HAVE_LAPACK */
#endif /* LOCALLYLINEAREMBEDDING_H_ */

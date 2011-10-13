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
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
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
 * internally for Lanzcos iterations (in case of only LAPACK is available)
 * and SUPERLU library for fast solving sparse equations stated by ARPACK 
 * is being used if available.
 *
 * This class also have capability of selecting k automatically in 
 * range between "k" and "max_k" in case if "auto_k" is true. This 
 * is being done using ternary search of minima of 
 * reconstruction error on the subset of features.
 */
class CLocallyLinearEmbedding: public CEmbeddingConverter
{
public:

	/** constructor */
	CLocallyLinearEmbedding();

	/** destructor */
	virtual ~CLocallyLinearEmbedding();

	/** apply preprocessor to features
	 * @param features
	 */
	virtual CSimpleFeatures<float64_t>* apply(CFeatures* features);

	/** setter for k parameter
	 * @param k k value
	 */
	void set_k(int32_t k);

	/** getter for k parameter
	 * @return m_k value
	 */
	int32_t get_k() const;

	/** setter for max_k parameter
	 * @param max_k max_k value
	 */
	void set_max_k(int32_t max_k);

	/** getter for max_k parameter
	 * @return m_max_k value
	 */
	int32_t get_max_k() const;

	/** setter for auto_k parameter
	 * @param auto_k auto_k value
	 */
	void set_auto_k(bool auto_k);

	/** getter for auto_k parameter
	 * @return m_auto_k value
	 */
	bool get_auto_k() const;

	/** setter for reconstruction shift parameter
	 * @param reconstruction_shift reconstruction shift value
	 */
	void set_reconstruction_shift(float64_t reconstruction_shift);

	/** getter for reconstruction shift parameter
	 * @return m_reconstruction_shift value
	 */
	float64_t get_reconstruction_shift() const;

	/** setter for nullspace shift parameter
	 * @param nullspace_shift nullsapce shift value
	 */
	void set_nullspace_shift(float64_t nullspace_shift);

	/** getter for nullspace shift parameter
	 * @return m_nullspace_shift value
	 */
	float64_t get_nullspace_shift() const;

	/** setter for use arpack parameter
	 * @param use_arpack use arpack value
	 */
	void set_use_arpack(bool use_arpack);

	/** getter for use arpack parameter
	 * @param use_arpack value
	 */
	bool get_use_arpack() const;

	/** get name */
	virtual const char* get_name() const;

	/// HELPERS
protected:

	/** default init */
	void init();

	/** constructs weight matrix 
	 * @param simple_features features to be used
	 * @param W_matrix weight matrix
	 * @param neighborhood_matrix matrix containing neighbor idxs
	 */
	virtual SGMatrix<float64_t> construct_weight_matrix(CSimpleFeatures<float64_t>* simple_features,float64_t* W_matrix, 
                                                            SGMatrix<int32_t> neighborhood_matrix);

	/** finds null space of given matrix 
	 * @param matrix given matrix
	 * @param dimension dimension of null space to be computed
	 * @return null-space approximation feature matrix
	 */
	SGMatrix<float64_t> find_null_space(SGMatrix<float64_t> matrix, int dimension);

	/** constructs neighborhood matrix by distance
	 * @param distance_matrix distance matrix to be used
	 * @param k number of neighbors
	 * @return matrix containing indexes of neighbors of i-th vector in i-th column
	 */
	SGMatrix<int32_t> get_neighborhood_matrix(SGMatrix<float64_t> distance_matrix, int32_t k);

	/** estimates k using ternary search 
	 * @param simple_features simple features to use
	 * @param neighborhood_matrix matrix containing indexes of neighbors for every vector
	 * @return optimal k (in means of reconstruction error)
	 */
	int32_t estimate_k(CSimpleFeatures<float64_t>* simple_features, SGMatrix<int32_t> neighborhood_matrix);

	/** computes reconstruction error using subset of given features
	 * @param k
	 * @param dim
	 * @param N
	 * @param feature_matrix
	 * @param z_matrix
	 * @param covariance_matrix
	 * @param resid_vector
	 * @param id_vector
	 * @param neighborhood_matrix
	 * @return residual sum
	 */
	float64_t compute_reconstruction_error(int32_t k, int dim, int N, float64_t* feature_matrix,
	                                       float64_t* z_matrix, float64_t* covariance_matrix,
	                                       float64_t* resid_vector, float64_t* id_vector,
	                                       SGMatrix<int32_t> neighborhood_matrix);

	/// FIELDS
protected:

	/** number of neighbors */
	int32_t m_k;

	/** maximum number of neighbors */
	int32_t m_max_k;

	/** regularization shift of reconstruction step */
	float64_t m_reconstruction_shift;

	/** regularization shift of nullspace finding step */
	float64_t m_nullspace_shift;

	/** whether use arpack or not */
	bool m_use_arpack;

	/** whether use automatic k or not */
	bool m_auto_k;

	/// THREADS
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

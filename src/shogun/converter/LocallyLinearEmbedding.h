/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LOCALLYLINEAREMBEDDING_H_
#define LOCALLYLINEAREMBEDDING_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/distance/Distance.h>
#include <shogun/converter/libedrt.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class LocallyLinearEmbedding (part of the Efficient
 * Dimensionality Reduction Toolkit) used to embed
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
 * the mean reconstruction error. The reconstruction error is
 * considered to have only one global minimum in this mode.
 *
 * It is optimized with alignment formulation as described in
 *
 * Zhao, D. (2006).
 * Formulating LLE using alignment technique.
 * Pattern Recognition, 39(11), 2233-2235.
 * Retrieved from http://linkinghub.elsevier.com/retrieve/pii/S0031320306002160
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','lle',k);
 *
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
	virtual CFeatures* apply(CFeatures* features);

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
	 * @return use_arpack value
	 */
	bool get_use_arpack() const;

	/** setter for use superlu parameter
	 * @param use_arpack use arpack value
	 */
	void set_use_superlu(bool use_superlu);

	/** getter for use superlu parameter
	 * @return use_arpack value
	 */
	bool get_use_superlu() const;

	/** get name */
	virtual const char* get_name() const;

	/// HELPERS
protected:

	/** default init */
	void init();

	static float64_t compute_kernel(int32_t i, int32_t j, const void* user_data);
	
	virtual const edrt_method_t get_edrt_method() const
	{
		return KERNEL_LOCALLY_LINEAR_EMBEDDING;
	}

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

	/** whether to use ARPACK or not */
	bool m_use_arpack;

	/** whether to use SuperLU or not */
	bool m_use_superlu;

	/** whether to use automatic k selection or not */
	bool m_auto_k;

};
}

#endif /* HAVE_LAPACK */
#endif /* LOCALLYLINEAREMBEDDING_H_ */

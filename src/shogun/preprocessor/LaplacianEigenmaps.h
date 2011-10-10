/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LAPLACIANEIGENMAPS_H_
#define LAPLACIANEIGENMAPS_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief the class LaplacianEigenmaps used to preprocess
 * data using Laplacian Eigenmaps algorithm as described in:
 *
 * Belkin, M., & Niyogi, P. (2002). 
 * Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering. 
 * Science, 14, 585-591. MIT Press. 
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.9400&rep=rep1&type=pdf
 *
 * Note that the algorithm is very sensitive to the heat distribution coefficient
 * and number of neighbors in the nearest neighbor graph. No connectivity check
 * is provided, so the preprocessor will not produce reasonable embeddings if the k value
 * makes a graph that is not connected. 
 *
 * This implementation is not parallel due to performance issues. Generalized 
 * eigenproblem is the bottleneck for this algorithm.
 *
 * Solving of generalized eigenproblem involves LAPACK DSYGVX routine
 * and requires extra memory for right-hand side matrix storage. 
 * If ARPACK is available then DSAUPD/DSEUPD is used with no extra 
 * memory usage. 
 *
 */
class CLaplacianEigenmaps: public CDimensionReductionPreprocessor<float64_t>
{
public:

	/** constructor */
	CLaplacianEigenmaps();

	/** destructor */
	virtual ~CLaplacianEigenmaps();

	/** init
	 * @param features
	 */
	virtual bool init(CFeatures* features);

	/** cleanup
	 *
	 */
	virtual void cleanup();

	/** apply preprocessor to feature matrix
	 * @param features
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preproc to feature vector
	 * @param vector
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

	/** setter for K parameter
	 * @param k k value
	 */
	void set_k(int32_t k);

	/** getter for K parameter
	 * @return k value
	 */
	int32_t get_k() const;

	/** setter for TAU parameter
	 * @param tau tau value
	 */
	void set_tau(float64_t tau);
	
	/** getter for TAU parameter
	 * @return tau value
	 */
	float64_t get_tau() const;

	/** get name */
	virtual const char* get_name() const;

	/** get type */
	virtual EPreprocessorType get_type() const;

protected:

	/** init */
	void init();

protected:

	/** number of neighbors */
	int32_t m_k;

	/** tau parameter of heat distribution */
	float64_t m_tau;

};
}

#endif /* HAVE_LAPACK */
#endif /* LAPLACIANEIGENMAPS_H_ */

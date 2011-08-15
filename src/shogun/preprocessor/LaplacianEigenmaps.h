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
#define LAPLCAIANEIGENMAPS_H_
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;

class CDistance;

/** @brief the class LaplacianEigenmaps used to preprocess
 * data using Laplacian Eigenmaps algorithm described in
 *
 * Belkin, M., & Niyogi, P. (2002). 
 * Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering. 
 * Science, 14, 585-591. MIT Press. 
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.9400&rep=rep1&type=pdf
 *
 */
class CLaplacianEigenmaps: public CDimensionReductionPreprocessor
{
public:

	/** constructor */
	CLaplacianEigenmaps();

	/** destructor */
	virtual ~CLaplacianEigenmaps();

	/** init
	 * @param data feature vectors for preproc
	 */
	virtual bool init(CFeatures* features);

	/** cleanup
	 *
	 */
	virtual void cleanup();

	/** apply preproc to feature matrix
	 *
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preproc to feature vector
	 *
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

	/** setter for K parameter
	 * @param k k
	 */
	void inline set_k(int32_t k)
	{
		m_k = k;
	}

	/** getter for K parameter
	 * @return k value
	 */
	int32_t inline get_k()
	{
		return m_k;
	}

	/** setter for TAU parameter
	 * @param tau tau
	 */
	void inline set_tau(float64_t tau)
	{
		m_tau = tau;
	}
	
	/** getter for TAU parameter
	 * @return tau value
	 */
	float64_t inline get_tau()
	{
		return m_tau;
	}

	/** get name */
	virtual inline const char* get_name() const { return "LaplacianEigenmaps"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_LAPLACIANEIGENMAPS; };

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
#endif /* LOCALLYLINEAREMBEDDING_H_ */

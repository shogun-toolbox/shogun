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
#ifdef HAVE_LAPACK
#include "preprocessor/DimensionReductionPreprocessor.h"
#include "features/Features.h"
#include "distance/Distance.h"

namespace shogun
{

class CFeatures;

class CDistance;

/** @brief the class LocallyLinearEmbedding used to preprocess
 *  data using Locally Linear Embedding algorithm described in
 *
 *	Saul, L. K., Ave, P., Park, F., & Roweis, S. T. (2001).
 *	An Introduction to Locally Linear Embedding. Available from, 290(5500), 2323-2326.
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

	/** get name */
	virtual inline const char* get_name() const { return "LocallyLinearEmbedding"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_LOCALLYLINEAREMBEDDING; };

protected:

	/* number of neighbors */
	int32_t m_k;

};
}

#endif /* HAVE_LAPACK */
#endif /* LOCALLYLINEAREMBEDDING_H_ */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LOCALLYLINEAREMBEDDING_H_
#define LOCALLYLINEAREMBEDDING_H_
#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class LocallyLinearEmbedding used to embed
 * data using Locally Linear Embedding algorithm described in
 *
 * Saul, L. K., Ave, P., Park, F., & Roweis, S. T. (2001).
 * An Introduction to Locally Linear Embedding. Available from, 290(5500), 2323-2326.
 * Retrieved from:
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.123.7319&rep=rep1&type=pdf
 *
 * It is optimized with alignment formulation as described in
 *
 * Zhao, D. (2006).
 * Formulating LLE using alignment technique.
 * Pattern Recognition, 39(11), 2233-2235.
 * Retrieved from http://linkinghub.elsevier.com/retrieve/pii/S0031320306002160
 *
 * Uses implementation from the Tapkee library.
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

	/** get name */
	virtual const char* get_name() const;

	/// HELPERS
protected:

	/** default init */
	void init();

	/// FIELDS
protected:

	/** number of neighbors */
	int32_t m_k;

	/** regularization shift of reconstruction step */
	float64_t m_reconstruction_shift;

	/** regularization shift of nullspace finding step */
	float64_t m_nullspace_shift;

};
}

#endif /* HAVE_EIGEN3 */
#endif /* LOCALLYLINEAREMBEDDING_H_ */

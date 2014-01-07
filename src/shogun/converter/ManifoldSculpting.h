/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Vladyslav S. Gorbatiuk
 * Copyright (C) 2011-2013 Vladyslav S. Gorbatiuk
 */

#ifndef MANIFOLDSCULPTING_H_
#define MANIFOLDSCULPTING_H_
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <converter/EmbeddingConverter.h>
#include <features/Features.h>

namespace shogun
{

/** @brief class CManifoldSculpting used to embed
 * data using manifold sculpting embedding algorithm.
 *
 * Uses implementation from the Tapkee library.
 *
 */
class CManifoldSculpting : public CEmbeddingConverter
{
public:

	/** constructor */
	CManifoldSculpting();

	/** destructor */
	virtual ~CManifoldSculpting();

	/** get name */
	virtual const char* get_name() const;

	/** apply preprocessor to features
	 *
	 * @param features features to embed
	 */
	virtual CFeatures* apply(CFeatures* features);

	/** setter for the k
	 *
	 * @param k the number of neighbors
	 */
	void set_k(const int32_t k);

	/** getter for the number of neighbors
	 *
	 * @return the number of neighbors k
	 */
	int32_t get_k() const;

	/** setter for squishing_rate
	 *
	 * @param squishing_rate the squishing rate
	 */
	void set_squishing_rate(const float64_t squishing_rate);

	/** getter for squishing_rate
	 *
	 * @return squishing_rate
	 */
	float64_t get_squishing_rate() const;

	/** setter for the maximum number of iterations
	 *
	 * @param max_iteration the maximum number of iterations
	 */
	void set_max_iteration(const int32_t max_iteration);

	/** getter for the maximum number of iterations
	 *
	 * @return the maximum number of iterations
	 */
	int32_t get_max_iteration() const;

private:

	/** default init */
	void init();

private:

	/** k - number of neighbors */
	float64_t m_k;

	/** squishing_rate */
	float64_t m_squishing_rate;

	/** max_iteration - the maximum number of algorithm's
	 * iterations
	 */
	float64_t m_max_iteration;

}; /* class CManifoldSculpting */

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */
#endif /* MANIFOLDSCULPTING_H_ */

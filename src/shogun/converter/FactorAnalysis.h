/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2011-2013 Fernando J. Iglesias Garcia
 */

#ifndef FACTOR_ANALYSIS_H_
#define FACTOR_ANALYSIS_H_
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <converter/EmbeddingConverter.h>
#include <features/Features.h>

namespace shogun
{

/** @brief class Factor Analysis used to embed
 * data using Factor Analysis algorithm.
 *
 * Uses implementation from the Tapkee library.
 *
 */
class CFactorAnalysis : public CEmbeddingConverter
{
public:

	/** constructor */
	CFactorAnalysis();

	/** destructor */
	virtual ~CFactorAnalysis();

	/** get name */
	virtual const char* get_name() const;

	/** apply preprocessor to features
	 *
	 * @param features features to embed
	 */
	virtual CFeatures* apply(CFeatures* features);

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

	/** setter for epsilon, parameter used to check for convergence
	 *
	 * @param epsilon convergence parameter
	 */
	void set_epsilon(const float64_t epsilon);

	/** getter for epsilon, parameter used to check for convergence
	 *
	 * @return value of the convergence parameter
	 */
	float64_t get_epsilon() const;

private:

	/** default init */
	void init();

private:

	/** maximum number of iterations */
	int32_t m_max_iteration;

	/** convergence parameter */
	float64_t m_epsilon;

}; /* class CFactorAnalysis */

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */
#endif /* FACTOR_ANALYSIS_H_ */

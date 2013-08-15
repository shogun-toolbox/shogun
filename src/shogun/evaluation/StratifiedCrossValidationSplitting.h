/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __STRATIFIEDCROSSVALIDATIONSPLITTING_H_
#define __STRATIFIEDCROSSVALIDATIONSPLITTING_H_

#include <shogun/evaluation/SplittingStrategy.h>

namespace shogun
{

class CLabels;

/** @brief Implementation of stratified cross-validation on the base of
 * CSplittingStrategy. Produces subset index sets of equal size (at most one
 * difference) in which the label ratio is equal (at most one difference) to
 * the label ratio of the specified labels. Do not use for regression since it
 * may be impossible to distribute nice in that case
 */
class CStratifiedCrossValidationSplitting: public CSplittingStrategy
{
public:
	/** constructor */
	CStratifiedCrossValidationSplitting();

	/** constructor
	 *
	 * @param labels labels to be (possibly) used for splitting
	 * @param num_subsets desired number of subsets, the labels are split into
	 */
	CStratifiedCrossValidationSplitting(CLabels* labels, index_t num_subsets);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "StratifiedCrossValidationSplitting";
	}

	/** implementation of the stratified cross-validation splitting strategy */
	virtual void build_subsets();

	/** custom rng if using cross validation across different threads */
	CRandom * m_rng;
};
}

#endif /* __STRATIFIEDCROSSVALIDATIONSPLITTING_H_ */

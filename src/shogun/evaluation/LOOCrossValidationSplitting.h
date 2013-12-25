/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Saurabh Mahindre
 */

#ifndef __LOOCROSSVALIDATIONSPLITTING_H_
#define __LOOCROSSVALIDATIONSPLITTING_H_

#include <shogun/evaluation/SplittingStrategy.h>

namespace shogun
{

class CLabels;

/** @brief Implementation of Leave one out cross-validation on the base of
 * CSplittingStrategy. Produces subset index sets consisting of one element,for each label.
 */
class CLOOCrossValidationSplitting: public CSplittingStrategy
{
public:
	/** constructor */
	CLOOCrossValidationSplitting();

	/** constructor
	 *
	 * @param labels labels to be (possibly) used for splitting
	 */
	CLOOCrossValidationSplitting(CLabels* labels);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "LOOCrossValidationSplitting";
	}

	/** implementation of the Leave one out cross-validation splitting strategy */
	virtual void build_subsets();

	/** custom rng if using cross validation across different threads */
	CRandom * m_rng;
};
}

#endif /* __LOOCROSSVALIDATIONSPLITTING_H_ */



/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein, Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef __CROSSVALIDATIONSPLITTING_H_
#define __CROSSVALIDATIONSPLITTING_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{

class Labels;

/** @brief Implementation of normal cross-validation on the base of
 * SplittingStrategy. Produces subset index sets of equal size (at most one
 * difference)
 */
class CrossValidationSplitting: public RandomMixin<SplittingStrategy>
{
public:
	/** constructor */
	CrossValidationSplitting();

	/** constructor
	 *
	 * @param labels labels to be (possibly) used for splitting
	 * @param num_subsets desired number of subsets, the labels are split into
	 */
	CrossValidationSplitting(std::shared_ptr<Labels> labels, index_t num_subsets);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "CrossValidationSplitting";
	}

	/** implementation of the standard cross-validation splitting strategy */
	virtual void build_subsets();
};
}

#endif /* __CROSSVALIDATIONSPLITTING_H_ */

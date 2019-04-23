/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Thoralf Klein, Yuyu Zhang
 */

#ifndef __STRATIFIEDCROSSVALIDATIONSPLITTING_H_
#define __STRATIFIEDCROSSVALIDATIONSPLITTING_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{

class Labels;

/** @brief Implementation of stratified cross-validation on the base of
 * SplittingStrategy. Produces subset index sets of equal size (at most one
 * difference) in which the label ratio is equal (at most one difference) to
 * the label ratio of the specified labels. Do not use for regression since it
 * may be impossible to distribute nice in that case
 */
class StratifiedCrossValidationSplitting: public RandomMixin<SplittingStrategy>
{
public:
	/** constructor */
	StratifiedCrossValidationSplitting();

	/** constructor
	 *
	 * @param labels labels to be (possibly) used for splitting
	 * @param num_subsets desired number of subsets, the labels are split into
	 */
	StratifiedCrossValidationSplitting(std::shared_ptr<Labels> labels, index_t num_subsets);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "StratifiedCrossValidationSplitting";
	}

	/** implementation of the stratified cross-validation splitting strategy */
	virtual void build_subsets();

protected:
	/* check for "stupid" combinations of label numbers and num_subsets.
	 * if there are of a class less labels than num_subsets, the class will not
	 * appear in every subset, leading to subsets of only one class in the
	 * extreme case of a two class labeling. */

	void check_labels() const;
};
}

#endif /* __STRATIFIEDCROSSVALIDATIONSPLITTING_H_ */

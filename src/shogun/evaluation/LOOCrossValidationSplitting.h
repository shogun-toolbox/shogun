/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Yuyu Zhang
 */

#ifndef __LOOCROSSVALIDATIONSPLITTING_H_
#define __LOOCROSSVALIDATIONSPLITTING_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/CrossValidationSplitting.h>

namespace shogun
{
/** @brief Implementation of Leave one out cross-validation on the base of
 * CrossValidationSplitting. Produces subset index sets consisting of one element,for each label.
 */
class LOOCrossValidationSplitting: public CrossValidationSplitting
{
public:
	/** constructor */
	LOOCrossValidationSplitting();

	/** constructor
	 *
	 * @param labels labels to be (possibly) used for splitting
	 */
	LOOCrossValidationSplitting(const std::shared_ptr<Labels>& labels);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "LOOCrossValidationSplitting";
	}

};
}

#endif /* __LOOCROSSVALIDATIONSPLITTING_H_ */



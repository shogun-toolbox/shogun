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
 * CCrossValidationSplitting. Produces subset index sets consisting of one element,for each label.
 */
class SHOGUN_EXPORT CLOOCrossValidationSplitting: public CCrossValidationSplitting
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

};
}

#endif /* __LOOCROSSVALIDATIONSPLITTING_H_ */



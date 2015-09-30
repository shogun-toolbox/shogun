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

#include <shogun/lib/config.h>

#include <shogun/evaluation/CrossValidationSplitting.h>

namespace shogun
{
/** @brief Implementation of Leave one out cross-validation on the base of
 * CCrossValidationSplitting. Produces subset index sets consisting of one element,for each label.
 */
class CLOOCrossValidationSplitting: public CCrossValidationSplitting
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



/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _ARGMAX_FUNCTION__H__
#define _ARGMAX_FUNCTION__H__

#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

class CArgMaxFunction;

/** output of the argmax function */
struct CResultSet : public CSGObject
{
	/** joint feature vector for the given truth */
	SGVector< float64_t > psi_truth;

	/** joint feature vector for the prediction */
	SGVector< float64_t > psi_pred;

	/** corresponding score */
	float64_t score;

	/** delta loss for the prediction vs. truth */
	float64_t delta;
};

/** @brief Class CArgMaxFunction is the case class of all the argmax
 * function used in Structured Output (SO) learning. This function
 * computes the output predicted by a classifier for any input.
 *
 */
class CArgMaxFunction : public CSGObject
{

	public:
		/** default constructor */
		CArgMaxFunction();

		/** destructor */
		virtual ~CArgMaxFunction();

		/** obtains the argmax
		 *
		 * @param features data in the input space \f$\mathcal{X}\f$
		 * @param feat_idx index of the feature to compute the argmax
		 * @param labels data in the output space \f$\mathcal{Y}\f$
		 * @param w weight vector
		 *
		 * @return structure with the predicted output
		 */
		virtual CResultSet* argmax(CFeatures* features, int32_t feat_idx, CStructuredLabels* labels, SGVector< float64_t> w) = 0;

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "ArgMaxFunction"; }

}; /* CArgMaxFunction */

} /* namespace shogun */

#endif /* _ARGMAX_FUNCTION__H__ */

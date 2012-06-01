/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTURED_LOSS_FUNCTION__H__
#define _STRUCTURED_LOSS_FUNCTION__H__

#include <shogun/base/SGObject.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/StructuredData.h>

namespace shogun
{

/**
 * @brief Class CStructuredLossFunction is the base class of all the 
 * loss functions used in Structured Output (SO) learning. These 
 * functions are commonly represented as 
 *
 * \f[
 * \Delta(y_{\text{true}}, y_{\text{pred}})
 * \f]
 *
 * The structured loss function computes the distance or difference 
 * between a label (structured) given by a classifier (e.g. SO-SVM)
 * and the ground truth.
 */
class CStructuredLossFunction : public CSGObject
{

	public:
		/** default constructor */
		CStructuredLossFunction();
		
		/** destructor */
		virtual ~CStructuredLossFunction();

		/** computes f$\Delta(y_{\text{true}}, y_{\text{pred}})\f$
		 *
		 * @param labels true labels
		 * @param ytrue_idx index of the true label in labels
		 * @param ypred the predicted label
		 *
		 * @return loss value
		 */
		virtual float64_t loss(CStructuredLabels* labels, int32_t ytrue_idx, CStructuredData ypred) = 0;

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "StructuredLossFunction"; }

}; /* CStructuredLossFunction */

} /* namespace shogun */

#endif /* _STRUCTURED_LOSS_FUNCITON__H__ */

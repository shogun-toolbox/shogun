/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTUREDOUTPUTMACHINE_H__
#define _STRUCTUREDOUTPUTMACHINE_H__

#include <shogun/so/StructuredLoss.h>
#include <shogun/so/StructuredModel.h>

namespace shogun
{

/** TODO doc */
class CStructuredOutputMachine : CMachine
{
	protected:
		/** feature vectors */
		CFeatures* m_features;

		/** the model that contains the application dependent modules */
		CStructuredModel* m_model;

		/** the general loss function */
		CStructuredLoss* m_loss;

}; /* class CStructuredOutputMachine */

} /* namespace shogun */

#endif /* _STRUCTUREDOUTPUTMACHINE_H__ */

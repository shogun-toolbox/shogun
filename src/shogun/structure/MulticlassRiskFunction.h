/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#ifndef _MULTICLASSRISK_FUNCTION__H__
#define _MULTICLASSRISK_FUNCITON__H__

#include <shogun/structure/RiskFunction.h>

namespace shogun
{

/** @brief Class CMulticlassRiskFunction
 *
 */
class CMulticlassRiskFunction : public CRiskFunction
{
	public:
		/** default constructor  */
		CMulticlassRiskFunction();

		/** destructor  */
		~CMulticlassRiskFunction();

		virtual void risk(void* data, float64_t* R, float64_t* subgrad, float64_t* W);

		virtual const char* get_name() const { return "MulticlassRiskFunction"; }

}; /* CMulticlassRiskFunction */

}

#endif /* _MULTICLASSTISK_FUNCTION__H__ */

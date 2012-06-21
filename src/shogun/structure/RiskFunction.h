/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#ifndef _RISK_FUNCTION__H__
#define _RISK_FUNCTION__H__

#include <shogun/base/SGObject.h>

namespace shogun
{

/** @brief Class CRiskFunction TODO
 *
 */
class CRiskFunction : public CSGObject
{
	public:
		/** default constructor */
		CRiskFunction();

		/** destructor */
		virtual ~CRiskFunction();

		/** computes the value of the risk function and sub-gradient at given point
		 *
		 */
		virtual void risk(void* data, float64_t* R, float64_t* subgrad, float64_t* W) = 0;

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "RiskFunction"; }

}; /* CRiskFunction */

}

#endif /* _RISK_FUNCTION__H__  */

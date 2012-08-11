/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef DIRECTOR_RISK_FUNCTION_H_
#define DIRECTOR_RISK_FUNCTION_H_

#include <shogun/structure/RiskFunction.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

/** @brief Class CDirectorRiskFunction TODO
 *
 */
class CDirectorRiskFunction : public CRiskFunction
{
	public:
		/** default constructor */
		CDirectorRiskFunction();

		/** destructor */
		virtual ~CDirectorRiskFunction();

		/** computes the value of the risk function and sub-gradient at given point
		 *
		 */
		virtual void risk(void* data, float64_t* R, float64_t* subgrad, float64_t* W);

		/** computes the value of the risk function and sub-gradient at given point
		 * director interface
		 */
		virtual void risk_directed(CDotFeatures* features, CLabels* labels, const SGVector<float64_t> R,
		                           const SGVector<float64_t> subgrad, const SGVector<float64_t> W);

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "DirectorRiskFunction"; }

}; /* CDirectorRiskFunction */
}
#endif /* DIRECTOR_RISK_FUNCTION_H_ */

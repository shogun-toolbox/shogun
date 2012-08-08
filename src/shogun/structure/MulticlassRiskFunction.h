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
#include <shogun/structure/MulticlassSOLabels.h>
#include <shogun/features/DotFeatures.h>

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

		/** function computing risk and subgradient at given point W */
		virtual void risk(void* data, float64_t* R, float64_t* subgrad, float64_t* W, TMultipleCPinfo* info=0);

		virtual const char* get_name() const { return "MulticlassRiskFunction"; }

}; /* CMulticlassRiskFunction */

/** @brief Class CMulticlassRiskData
 *
 */
class CMulticlassRiskData : public CRiskData
{
	public:
		/** default constructor */
		CMulticlassRiskData();

		/** constructor */
		CMulticlassRiskData(CDotFeatures *X, CMulticlassSOLabels *y, uint32_t w_dim, uint32_t nFeatures);

		/** destructor */
		~CMulticlassRiskData();

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "MulticlassRiskData"; }

	public:
		CDotFeatures*           m_X;
		CMulticlassSOLabels*    m_y;
};

}

#endif /* _MULTICLASSTISK_FUNCTION__H__ */

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
#define IGNORE_IN_CLASSLIST

/**
 * \struct TMultipleCPinfo
 * Multiple cutting plane models helper
 */
IGNORE_IN_CLASSLIST struct TMultipleCPinfo {
	/** where this portion of data starts */
	uint32_t from;
	/** how many examples belong to this portion of data */
	uint32_t N;
};

/** @brief Class CRiskFunction interface for Structured Output Risk functions
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
		 * @param data		User data passed to risk function
		 * @param R			Value of the computed risk at given point W
		 * @param subgrad	Subgradient computed at given point W
		 * @param W			Given weight vector
		 * @param info		Helper info for multiple cutting plane models algorithm
		 */
		virtual void risk(
				void* data,
				float64_t* 			R,
				float64_t* 			subgrad,
				float64_t* 			W,
				TMultipleCPinfo* 	info=0) = 0;

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "RiskFunction"; }
}; /* CRiskFunction */

/** @brief CRiskData interface for user data passed to risk function
 *
 */
class CRiskData : public CSGObject
{
	public:
		/** default constructor */
		CRiskData();

		/** constructor
		 *
		 * @param w_dim		Dimension of the weight vector W
		 * @param nFeatures	Overall number of features
		 */
		CRiskData(uint32_t w_dim, uint32_t nFeatures);

		/** destructor */
		~CRiskData();

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "RiskData"; }

	public:
		/** weight vector dimension */
		uint32_t m_w_dim;
		/** number of features */
		uint32_t m_nFeatures;
};

}

#endif /* _RISK_FUNCTION__H__  */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *  Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef _DIRECTOR_RISK_FUNCTION__H__
#define _DIRECTOR_RISK_FUNCITON__H__

#include <shogun/structure/RiskFunction.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

/** @brief Class CDirectorRiskFunction implements risk function
 *
 *	\sa CRiskFunction
 */
class CDirectorRiskFunction : public CRiskFunction
{
	public:
		/** default constructor  */
		CDirectorRiskFunction();

		/** destructor  */
		~CDirectorRiskFunction();

		/** function computing risk and subgradient at given point W */
		virtual void risk(
				void* 				data,
				float64_t* 			R,
				float64_t* 			subgrad,
				float64_t* 			W,
				TMultipleCPinfo* 	info=0);

		/** computes the value of the risk function and sub-gradient at given point
		 *  director inteface
		 */
		virtual void risk_directed(
				CDotFeatures* 				features,
				CLabels* 					labels,
				SGVector< float64_t > 		R,
				SGVector< float64_t > 		subgrad,
				const SGVector< float64_t > W);

		virtual const char* get_name() const { return "DirectorRiskFunction"; }

}; /* CMulticlassRiskFunction */

/** @brief Class CDirectorRiskData
 *
 * \sa CRiskData
 */
class CDirectorRiskData : public CRiskData
{
	public:
		/** default constructor */
		CDirectorRiskData();

		/** constructor
		 *
		 * @param X			Features
		 * @param y			Labels
		 * @param w_dim		Dimension of the weight vector W
		 * @param nFeatures	Number of features
		 */
		CDirectorRiskData(
				CDotFeatures* 	X,
				CLabels*        y,
				uint32_t 		w_dim,
				uint32_t 		nFeatures);

		/** destructor */
		~CDirectorRiskData();

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "DirectorRiskData"; }

	public:
		/** m_X */
		CDotFeatures*   m_X;
		/** m_y */
		CLabels*        m_y;
};

}

#endif /* _DIRECTOR_RISK_FUNCITON__H__ */

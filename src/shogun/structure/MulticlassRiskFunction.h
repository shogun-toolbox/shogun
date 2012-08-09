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

/** @brief Class CMulticlassRiskFunction implements risk function for multiclass SO-SVM
 *
 * Implementation of risk function for structured output multiclass SVM.
 *
 * The value of the risk is evaluated as
 *
 * \f[
 * R({\bf w}) = \sum_{i=1}^{m} \max_{y \in \mathcal{Y}} \left[ \ell(y_i, y) + \langle {\bf w}, \Psi(x_i, y) - \Psi(x_i, y_i)  \rangle  \right]
 * \f]
 *
 * The subgradient is by Danskin's theorem given as
 *
 * \f[
 * R'({\bf w}) = \sum_{i=1}^{m} \Psi(x_i, \hat{y}_i) - \Psi(x_i, y_i),
 * \f]
 *
 * where \f$ \hat{y}_i \f$ is the most violated label, i.e.
 *
 * \f[
 * \hat{y}_i = \arg\max_{y \in \mathcal{Y}} \left[ \ell(y_i, y) + \langle {\bf w}, \Psi(x_i, y)  \rangle \right]
 * \f]
 *
 *	\sa CRiskFunction
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

/** @brief Class CMulticlassRiskData provides risk data for multiclass SO-SVM
 *
 * \sa CRiskData
 */
class CMulticlassRiskData : public CRiskData
{
	public:
		/** default constructor */
		CMulticlassRiskData();

		/** constructor
		 *
		 * @param X			Features
		 * @param y			Labels
		 * @param w_dim		Dimension of the weight vector W
		 * @param nFeatures	Number of features
		 */
		CMulticlassRiskData(CDotFeatures *X, CMulticlassSOLabels *y, uint32_t w_dim, uint32_t nFeatures);

		/** destructor */
		~CMulticlassRiskData();

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "MulticlassRiskData"; }

	public:
		/** m_X */
		CDotFeatures*           m_X;
		/** m_y */
		CMulticlassSOLabels*    m_y;
};

}

#endif /* _MULTICLASSRISK_FUNCTION__H__ */

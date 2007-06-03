/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LPBOOST_H___
#define _LPBOOST_H___

#include "lib/config.h"
#ifdef USE_CPLEX

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "classifier/SparseLinearClassifier.h"

class CLPBoost : public CSparseLinearClassifier
{
	public:
		CLPBoost();
		virtual ~CLPBoost();

		virtual bool train();

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_LPBOOST;
		}
};
#endif //USE_CPLEX
#endif //_LPBOOST_H___

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LPM_H___
#define _LPM_H___

#include "lib/config.h"
#ifdef USE_CPLEX

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "classifier/SparseLinearClassifier.h"

class CLPM : public CSparseLinearClassifier
{
	public:
		CLPM();
		virtual ~CLPM();

		virtual bool train();

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_LPM;
		}

		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }
		inline bool get_bias_enabled() { return use_bias; }

		inline void set_epsilon(DREAL eps) { epsilon=eps; }
		inline DREAL get_epsilon() { return epsilon; }

	protected:
		DREAL C1;
		DREAL C2;
		bool use_bias;
		DREAL epsilon;
};
#endif //USE_CPLEX
#endif //_LPM_H___

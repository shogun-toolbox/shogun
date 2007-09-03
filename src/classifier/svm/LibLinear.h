/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBLINEAR_H___
#define _LIBLINEAR_H___

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"

enum LIBLINEAR_LOSS
{
	LR = 0,
	L2 = 1
};

class CLibLinear : public CSparseLinearClassifier
{
	public:
		CLibLinear(LIBLINEAR_LOSS loss);
		virtual ~CLibLinear();
		virtual bool train();
		virtual inline EClassifierType get_classifier_type() { return CT_LIBLINEAR; }
		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }
		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

		inline void set_epsilon(DREAL eps) { epsilon=eps; }
		inline DREAL get_epsilon() { return epsilon; }

		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }
		inline bool get_bias_enabled() { return use_bias; }

	protected:
		DREAL C1;
		DREAL C2;
		bool use_bias;
		DREAL epsilon;

		LIBLINEAR_LOSS loss;
};
#endif //HAVE_LAPACK
#endif //_LIBLINEAR_H___

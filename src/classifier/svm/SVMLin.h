/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SVMLIN_H___
#define _SVMLIN_H___

#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

class CSVMLin : public CSparseLinearClassifier
{
	public:
		CSVMLin();
		CSVMLin(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);
		virtual ~CSVMLin();

		virtual inline EClassifierType get_classifier_type() { return CT_SVMLIN; }
		virtual bool train();

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
		DREAL epsilon;

		bool use_bias;
};
#endif

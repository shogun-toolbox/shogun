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

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "classifier/LinearClassifier.h"
#include "classifier/svm/SVM.h"

class CSVMLin : public CLinearClassifier
{
	public:
		CSVMLin();
		virtual ~CSVMLin();

		inline EClassifierType get_classifier_type() { return CT_SVMLIN; }
		virtual bool train();

		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

	protected:
		DREAL C1;
		DREAL C2;
};
#endif

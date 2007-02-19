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

class CSVMLin : public CLinearClassifier, public CSVM
{
	public:
		CSVMLin();
		virtual ~CSVMLin();

		inline EClassifierType get_classifier_type() { return CT_SVMLIN; }
		virtual bool train();
};
#endif

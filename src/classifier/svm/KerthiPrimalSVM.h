/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERTHIPRIMALSVM_H___
#define _KERTHIPRIMALSVM_H___
#include "lib/common.h"
#include "classifier/Classifier.h"
#include "classifier/LinearClassifier.h"
#include "classifier/svm/SVM.h"
#include "lib/Cache.h"

class CKerthiPrimalSVM : public CLinearClassifier, public CSVM
{
	public:
		CKerthiPrimalSVM();
		virtual ~CKerthiPrimalSVM();
		virtual bool train();
};

#endif


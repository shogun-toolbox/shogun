/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MULTICLASSSVM_H___
#define _MULTICLASSSVM_H___

#include "lib/common.h"
#include "features/Features.h"
#include "classifier/svm/SVM.h"

class CSVM;

class CMultiClassSVM : public CSVM
{
	public:
		CMultiClassSVM();
		CMultiClassSVM(DREAL C, CKernel* k, CLabels* lab);

		virtual ~CMultiClassSVM();

		bool create_multiclass_svm(int num_classes);
		bool set_svm(INT num, CSVM* svm);
		void cleanup();

		virtual CLabels* classify(CLabels* labels=NULL);
		virtual DREAL classify_example(INT num);

	protected:
		INT num_svms;
		CSVM** svms;
};
#endif

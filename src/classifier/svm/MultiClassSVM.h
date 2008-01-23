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

enum EMultiClassSVM
{
	ONE_VS_REST,
	ONE_VS_ONE
};

class CMultiClassSVM : public CSVM
{
	public:
		CMultiClassSVM(EMultiClassSVM type);
		CMultiClassSVM(EMultiClassSVM type, DREAL C, CKernel* k, CLabels* lab);

		virtual ~CMultiClassSVM();

		bool create_multiclass_svm(int num_classes);
		bool set_svm(INT num, CSVM* svm);

		CSVM* get_svm(INT num)
		{
			ASSERT(m_num_svms>0);
			ASSERT(num>0 && num<m_num_svms);
			SG_REF(m_svms[num]);
			return m_svms[num];
		}

		INT inline get_num_svms()
		{
			return m_num_svms;
		}

		void cleanup();

		virtual CLabels* classify(CLabels* labels=NULL);
		virtual DREAL classify_example(INT num);

		CLabels* classify_one_vs_rest(CLabels* labels=NULL);
		DREAL classify_example_one_vs_rest(INT num);

		CLabels* classify_one_vs_one(CLabels* labels=NULL);
		DREAL classify_example_one_vs_one(INT num);

	protected:
		EMultiClassSVM multiclass_type;

		INT m_num_classes;
		INT m_num_svms;
		CSVM** m_svms;
};
#endif

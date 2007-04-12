/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "classifier/svm/MultiClassSVM.h"

CMultiClassSVM::CMultiClassSVM() : CSVM(0), num_svms(0), svms(NULL)
{
}

CMultiClassSVM::CMultiClassSVM(DREAL C, CKernel* k, CLabels* lab) : CSVM(C, k, lab)
{
}

CMultiClassSVM::~CMultiClassSVM()
{
	cleanup();
}

void CMultiClassSVM::cleanup()
{
	for (INT i=0; i<num_svms; i++)
		delete svms[i];
	delete[] svms;

	num_svms=0;
	svms=NULL;
}

bool CMultiClassSVM::create_multiclass_svm(INT num_classes)
{
	if (num_classes>0)
	{
		num_svms=num_classes;
		svms=new CSVM*[num_svms];
		memset(svms,0, num_svms*sizeof(CSVM*));

		if (svms)
			return true;
	}
	return false;
}

bool CMultiClassSVM::set_svm(INT num, CSVM* svm)
{
	if (num_svms>0 && num_svms>num && num>=0 && svm)
	{
		svms[num]=svm;
		return true;
	}
	return false;
}

CLabels* CMultiClassSVM::classify(CLabels* result)
{
	ASSERT(num_svms>0);

	if (!CKernelMachine::get_kernel())
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( (CKernelMachine::get_kernel()) &&
		 (CKernelMachine::get_kernel())->get_rhs() &&
		 (CKernelMachine::get_kernel())->get_rhs()->get_num_vectors())
	{
		INT num_vectors=get_kernel()->get_rhs()->get_num_vectors();

		if (!result)
			result=new CLabels(num_vectors);

		ASSERT(num_vectors == result->get_num_labels());

		ASSERT(result);
		CLabels** outputs=new CLabels*[num_svms];
		ASSERT(outputs);

		for (INT i=0; i<num_svms; i++)
		{
			ASSERT(svms[i]);
			svms[i]->set_kernel(get_kernel());
			svms[i]->set_labels(get_labels());
			outputs[i]=svms[i]->classify();
		}

		for (INT i=0; i<num_vectors; i++)
		{
			INT winner=0;
			DREAL max_out=outputs[0]->get_label(i);

			for (INT j=1; j<num_svms; j++)
			{
				DREAL out=outputs[j]->get_label(i);

				if (out>max_out)
				{
					winner=j;
					max_out=out;
				}
			}

			result->set_label(i, winner);
		}

		for (INT i=0; i<num_svms; i++)
			delete outputs[i];
		delete[] outputs;
	}

	return result;
}

DREAL CMultiClassSVM::classify_example(INT num)
{
	ASSERT(num_svms>0);
	DREAL* outputs=new DREAL[num_svms];
	ASSERT(outputs);

	INT winner=0;
	DREAL max_out=svms[0]->classify_example(num);

	for (INT i=1; i<num_svms; i++)
	{
		outputs[i]=svms[i]->classify_example(num);
		if (outputs[i]>max_out)
		{
			winner=i;
			max_out=outputs[i];
		}
	}
	delete[] outputs;

	return winner;
}

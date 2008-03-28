/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "classifier/svm/MultiClassSVM.h"

CMultiClassSVM::CMultiClassSVM(EMultiClassSVM type) : CSVM(0), multiclass_type(type), m_num_svms(0), m_svms(NULL)
{
}

CMultiClassSVM::CMultiClassSVM(EMultiClassSVM type, DREAL C, CKernel* k, CLabels* lab) : CSVM(C, k, lab), multiclass_type(type), m_num_svms(0), m_svms(NULL)
{
}

CMultiClassSVM::~CMultiClassSVM()
{
	cleanup();
}

void CMultiClassSVM::cleanup()
{
	for (INT i=0; i<m_num_svms; i++)
		delete m_svms[i];
	delete[] m_svms;

	m_num_svms=0;
	m_svms=NULL;
}

bool CMultiClassSVM::create_multiclass_svm(INT num_classes)
{
	if (num_classes>0)
	{
		m_num_classes=num_classes;

		if (multiclass_type==ONE_VS_REST)
			m_num_svms=num_classes;
		else if (multiclass_type==ONE_VS_ONE)
			m_num_svms=num_classes*(num_classes-1)/2;
		else
			SG_ERROR("unknown multiclass type\n");

		m_svms=new CSVM*[m_num_svms];
		if (m_svms)
		{
			memset(m_svms,0, m_num_svms*sizeof(CSVM*));
			return true;
		}
	}
	return false;
}

bool CMultiClassSVM::set_svm(INT num, CSVM* svm)
{
	if (m_num_svms>0 && m_num_svms>num && num>=0 && svm)
	{
		m_svms[num]=svm;
		return true;
	}
	return false;
}

CLabels* CMultiClassSVM::classify(CLabels* result)
{
	if (multiclass_type==ONE_VS_REST)
		return classify_one_vs_rest(result);
	else if (multiclass_type==ONE_VS_ONE)
		return classify_one_vs_one(result);
	else
		SG_ERROR("unknown multiclass type\n");

	return NULL;
}

CLabels* CMultiClassSVM::classify_one_vs_one(CLabels* result)
{
	ASSERT(m_num_svms>0);
	ASSERT(m_num_svms==m_num_classes*(m_num_classes-1)/2);


	if (!kernel)
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( kernel && kernel->get_rhs() && kernel->get_rhs()->get_num_vectors())
	{
		INT num_vectors=kernel->get_rhs()->get_num_vectors();

		if (!result)
			result=new CLabels(num_vectors);

		ASSERT(num_vectors == result->get_num_labels());

		ASSERT(result);
		CLabels** outputs=new CLabels*[m_num_svms];
		ASSERT(outputs);

		for (INT i=0; i<m_num_svms; i++)
		{
			SG_INFO("num_svms:%d svm[%d]=0x%0X\n", m_num_svms, i, m_svms[i]);
			ASSERT(m_svms[i]);
			m_svms[i]->set_kernel(kernel);
			m_svms[i]->set_labels(labels);
			outputs[i]=m_svms[i]->classify();
		}

		INT* votes=new INT[m_num_classes];
		ASSERT(votes);

		for (INT v=0; v<num_vectors; v++)
		{
			INT s=0;
			memset(votes, 0, sizeof(INT)*m_num_classes);

			for (INT i=0; i<m_num_classes; i++)
			{
				for (INT j=i+1; j<m_num_classes; j++)
				{
					if (outputs[s++]->get_label(v)>0)
						votes[i]++;
					else
						votes[j]++;
				}
			}

			INT winner=0;
			INT max_votes=votes[0];

			for (INT i=1; i<m_num_classes; i++)
			{
				if (votes[i]>max_votes)
				{
					max_votes=votes[i];
					winner=i;
				}
			}

			result->set_label(v, winner);
		}

		delete[] votes;

		for (INT i=0; i<m_num_svms; i++)
			delete outputs[i];
		delete[] outputs;
	}

	return result;
}

CLabels* CMultiClassSVM::classify_one_vs_rest(CLabels* result)
{
	ASSERT(m_num_svms>0);

	if (!kernel)
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( kernel && kernel->get_rhs() && kernel->get_rhs()->get_num_vectors())
	{
		INT num_vectors=kernel->get_rhs()->get_num_vectors();

		if (!result)
			result=new CLabels(num_vectors);

		ASSERT(num_vectors == result->get_num_labels());

		ASSERT(result);
		CLabels** outputs=new CLabels*[m_num_svms];
		ASSERT(outputs);

		for (INT i=0; i<m_num_svms; i++)
		{
			ASSERT(m_svms[i]);
			m_svms[i]->set_kernel(kernel);
			m_svms[i]->set_labels(get_labels());
			outputs[i]=m_svms[i]->classify();
		}

		for (INT i=0; i<num_vectors; i++)
		{
			INT winner=0;
			DREAL max_out=outputs[0]->get_label(i);

			for (INT j=1; j<m_num_svms; j++)
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

		for (INT i=0; i<m_num_svms; i++)
			delete outputs[i];
		delete[] outputs;
	}

	return result;
}

DREAL CMultiClassSVM::classify_example(INT num)
{
	if (multiclass_type==ONE_VS_REST)
		return classify_example_one_vs_rest(num);
	else if (multiclass_type==ONE_VS_ONE)
		return classify_example_one_vs_one(num);
	else
		SG_ERROR("unknown multiclass type\n");

	return 0;
}

DREAL CMultiClassSVM::classify_example_one_vs_rest(INT num)
{
	ASSERT(m_num_svms>0);
	DREAL* outputs=new DREAL[m_num_svms];
	ASSERT(outputs);

	INT winner=0;
	DREAL max_out=m_svms[0]->classify_example(num);

	for (INT i=1; i<m_num_svms; i++)
	{
		outputs[i]=m_svms[i]->classify_example(num);
		if (outputs[i]>max_out)
		{
			winner=i;
			max_out=outputs[i];
		}
	}
	delete[] outputs;

	return winner;
}

DREAL CMultiClassSVM::classify_example_one_vs_one(INT num)
{
	ASSERT(m_num_svms>0);
	ASSERT(m_num_svms==m_num_classes*(m_num_classes-1)/2);

	INT* votes=new INT[m_num_classes];
	ASSERT(votes);

	INT s=0;

	for (INT i=0; i<m_num_classes; i++)
	{
		for (INT j=i+1; j<m_num_classes; j++)
		{
			if (m_svms[s++]->classify_example(num)>0)
				votes[i]++;
			else
				votes[j]++;
		}
	}

	INT winner=0;
	INT max_votes=votes[0];

	for (INT i=1; i<m_num_classes; i++)
	{
		if (votes[i]>max_votes)
		{
			max_votes=votes[i];
			winner=i;
		}
	}

	delete[] votes;

	return winner;
}

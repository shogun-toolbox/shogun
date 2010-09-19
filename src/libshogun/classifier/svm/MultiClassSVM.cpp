/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Parameter.h"
#include "classifier/svm/MultiClassSVM.h"

using namespace shogun;

CMultiClassSVM::CMultiClassSVM(EMultiClassSVM type)
: CSVM(0), multiclass_type(type), m_num_svms(0), m_svms(NULL)
{
	init();
}

CMultiClassSVM::CMultiClassSVM(
	EMultiClassSVM type, float64_t C, CKernel* k, CLabels* lab)
: CSVM(C, k, lab), multiclass_type(type), m_num_svms(0), m_svms(NULL)
{
	init();
}

CMultiClassSVM::~CMultiClassSVM()
{
	cleanup();
}

void
CMultiClassSVM::init(void)
{
	m_parameters->add_int32((int32_t*) &multiclass_type,
							"multiclass_type",
							"Type of the MultiClassSVM.");
	m_parameters->add_int32(&m_num_classes,
							"num_classes",
							"Number of classes.");
	m_parameters->add_int32(&m_num_svms, "num_svms",
							"Number of SVMs.");
}

void CMultiClassSVM::cleanup()
{
	for (int32_t i=0; i<m_num_svms; i++)
		SG_UNREF(m_svms[i]);

	delete[] m_svms;
	m_num_svms=0;
	m_svms=NULL;
}

bool CMultiClassSVM::create_multiclass_svm(int32_t num_classes)
{
	if (num_classes>0)
	{
		cleanup();

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

bool CMultiClassSVM::set_svm(int32_t num, CSVM* svm)
{
	if (m_num_svms>0 && m_num_svms>num && num>=0 && svm)
	{
		SG_REF(svm);
		m_svms[num]=svm;
		return true;
	}
	return false;
}

CLabels* CMultiClassSVM::classify()
{
	if (multiclass_type==ONE_VS_REST)
		return classify_one_vs_rest();
	else if (multiclass_type==ONE_VS_ONE)
		return classify_one_vs_one();
	else
		SG_ERROR("unknown multiclass type\n");

	return NULL;
}

CLabels* CMultiClassSVM::classify_one_vs_one()
{
	ASSERT(m_num_svms>0);
	ASSERT(m_num_svms==m_num_classes*(m_num_classes-1)/2);
	CLabels* result=NULL;

	if (!kernel)
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( kernel && kernel->get_num_vec_lhs() && kernel->get_num_vec_rhs())
	{
		int32_t num_vectors=kernel->get_num_vec_rhs();

		result=new CLabels(num_vectors);
		SG_REF(result);

		ASSERT(num_vectors==result->get_num_labels());
		CLabels** outputs=new CLabels*[m_num_svms];

		for (int32_t i=0; i<m_num_svms; i++)
		{
			SG_INFO("num_svms:%d svm[%d]=0x%0X\n", m_num_svms, i, m_svms[i]);
			ASSERT(m_svms[i]);
			m_svms[i]->set_kernel(kernel);
			outputs[i]=m_svms[i]->classify();
		}

		int32_t* votes=new int32_t[m_num_classes];
		for (int32_t v=0; v<num_vectors; v++)
		{
			int32_t s=0;
			memset(votes, 0, sizeof(int32_t)*m_num_classes);

			for (int32_t i=0; i<m_num_classes; i++)
			{
				for (int32_t j=i+1; j<m_num_classes; j++)
				{
					if (outputs[s++]->get_label(v)>0)
						votes[i]++;
					else
						votes[j]++;
				}
			}

			int32_t winner=0;
			int32_t max_votes=votes[0];

			for (int32_t i=1; i<m_num_classes; i++)
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

		for (int32_t i=0; i<m_num_svms; i++)
			SG_UNREF(outputs[i]);
		delete[] outputs;
	}

	return result;
}

CLabels* CMultiClassSVM::classify_one_vs_rest()
{
	ASSERT(m_num_svms>0);
	CLabels* result=NULL;

	if (!kernel)
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( kernel && kernel->get_num_vec_lhs() && kernel->get_num_vec_rhs())
	{
		int32_t num_vectors=kernel->get_num_vec_rhs();

		result=new CLabels(num_vectors);
		SG_REF(result);

		ASSERT(num_vectors==result->get_num_labels());
		CLabels** outputs=new CLabels*[m_num_svms];

		for (int32_t i=0; i<m_num_svms; i++)
		{
			ASSERT(m_svms[i]);
			m_svms[i]->set_kernel(kernel);
			outputs[i]=m_svms[i]->classify();
		}

		for (int32_t i=0; i<num_vectors; i++)
		{
			int32_t winner=0;
			float64_t max_out=outputs[0]->get_label(i);

			for (int32_t j=1; j<m_num_svms; j++)
			{
				float64_t out=outputs[j]->get_label(i);

				if (out>max_out)
				{
					winner=j;
					max_out=out;
				}
			}

			result->set_label(i, winner);
		}

		for (int32_t i=0; i<m_num_svms; i++)
			SG_UNREF(outputs[i]);

		delete[] outputs;
	}

	return result;
}

float64_t CMultiClassSVM::classify_example(int32_t num)
{
	if (multiclass_type==ONE_VS_REST)
		return classify_example_one_vs_rest(num);
	else if (multiclass_type==ONE_VS_ONE)
		return classify_example_one_vs_one(num);
	else
		SG_ERROR("unknown multiclass type\n");

	return 0;
}

float64_t CMultiClassSVM::classify_example_one_vs_rest(int32_t num)
{
	ASSERT(m_num_svms>0);
	float64_t* outputs=new float64_t[m_num_svms];
	int32_t winner=0;
	float64_t max_out=m_svms[0]->classify_example(num);

	for (int32_t i=1; i<m_num_svms; i++)
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

float64_t CMultiClassSVM::classify_example_one_vs_one(int32_t num)
{
	ASSERT(m_num_svms>0);
	ASSERT(m_num_svms==m_num_classes*(m_num_classes-1)/2);

	int32_t* votes=new int32_t[m_num_classes];
	int32_t s=0;

	for (int32_t i=0; i<m_num_classes; i++)
	{
		for (int32_t j=i+1; j<m_num_classes; j++)
		{
			if (m_svms[s++]->classify_example(num)>0)
				votes[i]++;
			else
				votes[j]++;
		}
	}

	int32_t winner=0;
	int32_t max_votes=votes[0];

	for (int32_t i=1; i<m_num_classes; i++)
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

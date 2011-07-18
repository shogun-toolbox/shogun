/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/io/io.h>
#include <shogun/classifier/svm/MultiClassSVM.h>

using namespace shogun;

CMultiClassSVM::CMultiClassSVM(void)
: CSVM(0), multiclass_type(ONE_VS_REST), m_num_svms(0), m_svms(NULL)
{
	init();
}

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

void CMultiClassSVM::init()
{
	m_parameters->add((machine_int_t*) &multiclass_type,
					  "multiclass_type", "Type of MultiClassSVM.");
	m_parameters->add(&m_num_classes, "m_num_classes",
					  "Number of classes.");
	m_parameters->add_vector((CSGObject***) &m_svms,
							 &m_num_svms, "m_svms");
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

CLabels* CMultiClassSVM::apply()
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
			outputs[i]=m_svms[i]->apply();
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
			outputs[i]=m_svms[i]->apply();
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

float64_t CMultiClassSVM::apply(int32_t num)
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
	float64_t max_out=m_svms[0]->apply(num);

	for (int32_t i=1; i<m_num_svms; i++)
	{
		outputs[i]=m_svms[i]->apply(num);
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
			if (m_svms[s++]->apply(num)>0)
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

bool CMultiClassSVM::load(FILE* modelfl)
{
	bool result=true;
	char char_buffer[1024];
	int32_t int_buffer;
	float64_t double_buffer;
	int32_t line_number=1;
	int32_t svm_idx=-1;

	SG_SET_LOCALE_C;

	if (fscanf(modelfl,"%15s\n", char_buffer)==EOF)
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);
	else
	{
		char_buffer[15]='\0';
		if (strcmp("%MultiClassSVM", char_buffer)!=0)
			SG_ERROR( "error in multiclass svm file, line nr:%d\n", line_number);

		line_number++;
	}

	int_buffer=0;
	if (fscanf(modelfl," multiclass_type=%d; \n", &int_buffer) != 1)
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);

	if (!feof(modelfl))
		line_number++;

	if (int_buffer != multiclass_type)
		SG_ERROR("multiclass type does not match %ld vs. %ld\n", int_buffer, multiclass_type);

	int_buffer=0;
	if (fscanf(modelfl," num_classes=%d; \n", &int_buffer) != 1)
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);

	if (!feof(modelfl))
		line_number++;

	if (int_buffer < 2)
		SG_ERROR("less than 2 classes - how is this multiclass?\n");

	create_multiclass_svm(int_buffer);

	int_buffer=0;
	if (fscanf(modelfl," num_svms=%d; \n", &int_buffer) != 1)
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);

	if (!feof(modelfl))
		line_number++;

	if (m_num_svms != int_buffer)
		SG_ERROR("Mismatch in number of svms: m_num_svms=%d vs m_num_svms(file)=%d\n", m_num_svms, int_buffer);

	if (fscanf(modelfl," kernel='%s'; \n", char_buffer) != 1)
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);

	if (!feof(modelfl))
		line_number++;

	for (int32_t n=0; n<m_num_svms; n++)
	{
		svm_idx=-1;
		if (fscanf(modelfl,"\n%4s %d of %d\n", char_buffer, &svm_idx, &int_buffer)==EOF)
		{
			result=false;
			SG_ERROR( "error in svm file, line nr:%d\n", line_number);
		}
		else
		{
			char_buffer[4]='\0';
			if (strncmp("%SVM", char_buffer, 4)!=0)
			{
				result=false;
				SG_ERROR( "error in svm file, line nr:%d\n", line_number);
			}

			if (svm_idx != n)
				SG_ERROR("svm index mismatch n=%d, n(file)=%d\n", n, svm_idx);

			line_number++;
		}

		int_buffer=0;
		if (fscanf(modelfl,"numsv%d=%d;\n", &svm_idx, &int_buffer) != 2)
			SG_ERROR( "error in svm file, line nr:%d\n", line_number);

		if (svm_idx != n)
			SG_ERROR("svm index mismatch n=%d, n(file)=%d\n", n, svm_idx);

		if (!feof(modelfl))
			line_number++;

		SG_INFO("loading %ld support vectors for svm %d\n",int_buffer, svm_idx);
		CSVM* svm=new CSVM(int_buffer);

		double_buffer=0;

		if (fscanf(modelfl," b%d=%lf; \n", &svm_idx, &double_buffer) != 2)
			SG_ERROR( "error in svm file, line nr:%d\n", line_number);

		if (svm_idx != n)
			SG_ERROR("svm index mismatch n=%d, n(file)=%d\n", n, svm_idx);

		if (!feof(modelfl))
			line_number++;

		svm->set_bias(double_buffer);

		if (fscanf(modelfl,"alphas%d=[\n", &svm_idx) != 1)
			SG_ERROR( "error in svm file, line nr:%d\n", line_number);

		if (svm_idx != n)
			SG_ERROR("svm index mismatch n=%d, n(file)=%d\n", n, svm_idx);

		if (!feof(modelfl))
			line_number++;

		for (int32_t i=0; i<svm->get_num_support_vectors(); i++)
		{
			double_buffer=0;
			int_buffer=0;

			if (fscanf(modelfl,"\t[%lf,%d]; \n", &double_buffer, &int_buffer) != 2)
				SG_ERROR( "error in svm file, line nr:%d\n", line_number);

			if (!feof(modelfl))
				line_number++;

			svm->set_support_vector(i, int_buffer);
			svm->set_alpha(i, double_buffer);
		}

		if (fscanf(modelfl,"%2s", char_buffer) == EOF)
		{
			result=false;
			SG_ERROR( "error in svm file, line nr:%d\n", line_number);
		}
		else
		{
			char_buffer[3]='\0';
			if (strcmp("];", char_buffer)!=0)
			{
				result=false;
				SG_ERROR( "error in svm file, line nr:%d\n", line_number);
			}
			line_number++;
		}

		set_svm(n, svm);
	}

	svm_loaded=result;

	SG_RESET_LOCALE;
	return result;
}

bool CMultiClassSVM::save(FILE* modelfl)
{
	SG_SET_LOCALE_C;

	if (!kernel)
		SG_ERROR("Kernel not defined!\n");

	if (!m_svms || m_num_svms<1 || m_num_classes <=2)
		SG_ERROR("Multiclass SVM not trained!\n");

	SG_INFO( "Writing model file...");
	fprintf(modelfl,"%%MultiClassSVM\n");
	fprintf(modelfl,"multiclass_type=%d;\n", multiclass_type);
	fprintf(modelfl,"num_classes=%d;\n", m_num_classes);
	fprintf(modelfl,"num_svms=%d;\n", m_num_svms);
	fprintf(modelfl,"kernel='%s';\n", kernel->get_name());

	for (int32_t i=0; i<m_num_svms; i++)
	{
		CSVM* svm=m_svms[i];
		ASSERT(svm);
		fprintf(modelfl,"\n%%SVM %d of %d\n", i, m_num_svms-1);
		fprintf(modelfl,"numsv%d=%d;\n", i, svm->get_num_support_vectors());
		fprintf(modelfl,"b%d=%+10.16e;\n",i,svm->get_bias());

		fprintf(modelfl, "alphas%d=[\n", i);

		for(int32_t j=0; j<svm->get_num_support_vectors(); j++)
		{
			fprintf(modelfl,"\t[%+10.16e,%d];\n",
					svm->get_alpha(j), svm->get_support_vector(j));
		}

		fprintf(modelfl, "];\n");
	}

	SG_RESET_LOCALE;
	SG_DONE();
	return true ;
}

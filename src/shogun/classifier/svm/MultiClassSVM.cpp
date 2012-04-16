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
#include <shogun/io/SGIO.h>
#include <shogun/classifier/svm/MultiClassSVM.h>

using namespace shogun;

CMultiClassSVM::CMultiClassSVM()
	:CKernelMulticlassMachine(ONE_VS_REST_STRATEGY, NULL, new CSVM(0), NULL)
{
	init();
}

CMultiClassSVM::CMultiClassSVM(EMulticlassStrategy strategy)
	:CKernelMulticlassMachine(strategy, NULL, new CSVM(0), NULL)
{
	init();
}

CMultiClassSVM::CMultiClassSVM(
	EMulticlassStrategy strategy, float64_t C, CKernel* k, CLabels* lab)
	:CKernelMulticlassMachine(strategy, k, new CSVM(C, k, lab), lab)
{
	init();
}

CMultiClassSVM::~CMultiClassSVM()
{
	cleanup();
}

void CMultiClassSVM::init()
{
}

void CMultiClassSVM::cleanup()
{
	clear_machines();
}

bool CMultiClassSVM::create_multiclass_svm(int32_t num_classes)
{
	if (num_classes>0)
	{
		cleanup();

		int32_t num_svms=0;
		if (m_multiclass_strategy==ONE_VS_REST_STRATEGY)
			num_svms=num_classes;
		else if (m_multiclass_strategy==ONE_VS_ONE_STRATEGY)
			num_svms=num_classes*(num_classes-1)/2;
		else
			SG_ERROR("unknown multiclass strategy\n");

		m_machines = SGVector<CMachine *>(num_svms);
		for (index_t i=0; i<num_svms; ++i)
			m_machines[i]=NULL;

		return true;
	}
	return false;
}

bool CMultiClassSVM::set_svm(int32_t num, CSVM* svm)
{
	if (m_machines.vlen>0 && m_machines.vlen>num && num>=0 && svm)
	{
		SG_REF(svm);
		m_machines[num]=svm;
		return true;
	}
	return false;
}

CLabels* CMultiClassSVM::apply(CFeatures* data)
{
	if (is_data_locked())
	{
		SG_ERROR("CKernelMachine::apply(CFeatures*) cannot be called when "
				"data_lock was called before. Call data_unlock to allow.");
	}

	if (!m_kernel)
		SG_ERROR("No kernel assigned!\n");

	CFeatures* lhs=m_kernel->get_lhs();
	if (!lhs)
		SG_ERROR("%s: No left hand side specified\n", get_name());

	if (!lhs->get_num_vectors())
	{
		SG_ERROR("%s: No vectors on left hand side (%s). This is probably due to"
				" an implementation error in %s, where it was forgotten to set "
				"the data (m_svs) indices\n", get_name(),
				data->get_name());
	}

	m_kernel->init(lhs, data);
	SG_UNREF(lhs);

	return apply();
}

CLabels* CMultiClassSVM::apply()
{
	if (m_multiclass_strategy==ONE_VS_REST_STRATEGY)
		return classify_one_vs_rest();
	else if (m_multiclass_strategy==ONE_VS_ONE_STRATEGY)
		return classify_one_vs_one();
	else
		SG_ERROR("unknown multiclass strategy\n");

	return NULL;
}

CLabels* CMultiClassSVM::classify_one_vs_one()
{
	int32_t num_classes=m_labels->get_num_classes();
	ASSERT(m_machines.vlen>0);
	ASSERT(m_machines.vlen==num_classes*(num_classes-1)/2);
	CLabels* result=NULL;

	if (!m_kernel)
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return NULL;
	}

	if (!( m_kernel && m_kernel->get_num_vec_lhs() && m_kernel->get_num_vec_rhs()))
		return NULL;

	int32_t num_vectors=m_kernel->get_num_vec_rhs();

	result=new CLabels(num_vectors);
	SG_REF(result);

	ASSERT(num_vectors==result->get_num_labels());
	CLabels** outputs=SG_MALLOC(CLabels*, m_machines.vlen);

	for (int32_t i=0; i<m_machines.vlen; i++)
	{
		SG_INFO("num_svms:%d svm[%d]=0x%0X\n", m_machines.vlen, i, m_machines[i]);
		ASSERT(m_machines[i]);
		CSVM *the_svm=(CSVM *)m_machines[i];
		the_svm->set_kernel(m_kernel);
		outputs[i]=the_svm->apply();
	}

	int32_t* votes=SG_MALLOC(int32_t, num_classes);
	for (int32_t v=0; v<num_vectors; v++)
	{
		int32_t s=0;
		memset(votes, 0, sizeof(int32_t)*num_classes);

		for (int32_t i=0; i<num_classes; i++)
		{
			for (int32_t j=i+1; j<num_classes; j++)
			{
				if (outputs[s++]->get_label(v)>0)
					votes[i]++;
				else
					votes[j]++;
			}
		}

		int32_t winner=0;
		int32_t max_votes=votes[0];

		for (int32_t i=1; i<num_classes; i++)
		{
			if (votes[i]>max_votes)
			{
				max_votes=votes[i];
				winner=i;
			}
		}

		result->set_label(v, winner);
	}

	SG_FREE(votes);

	for (int32_t i=0; i<m_machines.vlen; i++)
		SG_UNREF(outputs[i]);
	SG_FREE(outputs);

	return result;
}

CLabels* CMultiClassSVM::classify_one_vs_rest()
{
	ASSERT(m_machines.vlen>0);
	CLabels* result=NULL;

	if (!m_kernel)
	{
		SG_ERROR("SVM can not proceed without kernel!\n");
		return NULL;
	}

	if ( m_kernel && m_kernel->get_num_vec_lhs() && m_kernel->get_num_vec_rhs())
	{
		int32_t num_vectors=m_kernel->get_num_vec_rhs();

		result=new CLabels(num_vectors);
		SG_REF(result);

		ASSERT(num_vectors==result->get_num_labels());
		CLabels** outputs=SG_MALLOC(CLabels*, m_machines.vlen);

		for (int32_t i=0; i<m_machines.vlen; i++)
		{
			ASSERT(m_machines[i]);
			CSVM *the_svm = (CSVM *)m_machines[i];
			the_svm->set_kernel(m_kernel);
			outputs[i]=the_svm->apply();
		}

		for (int32_t i=0; i<num_vectors; i++)
		{
			int32_t winner=0;
			float64_t max_out=outputs[0]->get_label(i);

			for (int32_t j=1; j<m_machines.vlen; j++)
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

		for (int32_t i=0; i<m_machines.vlen; i++)
			SG_UNREF(outputs[i]);

		SG_FREE(outputs);
	}

	return result;
}

float64_t CMultiClassSVM::apply(int32_t num)
{
	if (m_multiclass_strategy==ONE_VS_REST_STRATEGY)
		return classify_example_one_vs_rest(num);
	else if (m_multiclass_strategy==ONE_VS_ONE_STRATEGY)
		return classify_example_one_vs_one(num);
	else
		SG_ERROR("unknown multiclass strategy\n");

	return 0;
}

float64_t CMultiClassSVM::classify_example_one_vs_rest(int32_t num)
{
	ASSERT(m_machines.vlen>0);
	float64_t* outputs=SG_MALLOC(float64_t, m_machines.vlen);
	int32_t winner=0;
	float64_t max_out=get_svm(0)->apply(num);

	for (int32_t i=1; i<m_machines.vlen; i++)
	{
		outputs[i]=get_svm(i)->apply(num);
		if (outputs[i]>max_out)
		{
			winner=i;
			max_out=outputs[i];
		}
	}
	SG_FREE(outputs);

	return winner;
}

float64_t CMultiClassSVM::classify_example_one_vs_one(int32_t num)
{
	int32_t num_classes=m_labels->get_num_classes();
	ASSERT(m_machines.vlen>0);
	ASSERT(m_machines.vlen==num_classes*(num_classes-1)/2);

	SGVector<int32_t> votes(num_classes);

	/* set votes array to zero to prevent uninitialized values if class gets
	 * no votes */
	votes.set_const(0);
	int32_t s=0;

	for (int32_t i=0; i<num_classes; i++)
	{
		for (int32_t j=i+1; j<num_classes; j++)
		{
			/** TODO, this was never used before, make classify_on_vs_one()
			 * use this code, instead of having a duplicate copy down there */
			get_svm(s)->set_kernel(m_kernel);
			if (get_svm(s)->apply(num)>0)
				votes.vector[i]++;
			else
				votes.vector[j]++;

			s++;
		}
	}

	int32_t winner=0;
	int32_t max_votes=votes.vector[0];

	for (int32_t i=1; i<num_classes; i++)
	{
		if (votes.vector[i]>max_votes)
		{
			max_votes=votes.vector[i];
			winner=i;
		}
	}

	votes.destroy_vector();

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

	/*
	int_buffer=0;
	if (fscanf(modelfl," multiclass_type=%d; \n", &int_buffer) != 1)
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);

	if (!feof(modelfl))
		line_number++;

	if (int_buffer != multiclass_type)
		SG_ERROR("multiclass type does not match %ld vs. %ld\n", int_buffer, multiclass_type);
	*/

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

	if (m_machines.vlen != int_buffer)
		SG_ERROR("Mismatch in number of svms: m_num_svms=%d vs m_num_svms(file)=%d\n", m_machines.vlen, int_buffer);

	if (fscanf(modelfl," kernel='%s'; \n", char_buffer) != 1)
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);

	if (!feof(modelfl))
		line_number++;

	for (int32_t n=0; n<m_machines.vlen; n++)
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

	svm_proto()->svm_loaded=result;

	SG_RESET_LOCALE;
	return result;
}

bool CMultiClassSVM::save(FILE* modelfl)
{
	SG_SET_LOCALE_C;

	if (!m_kernel)
		SG_ERROR("Kernel not defined!\n");

	if (m_machines.vlen<1)
		SG_ERROR("Multiclass SVM not trained!\n");

	SG_INFO( "Writing model file...");
	fprintf(modelfl,"%%MultiClassSVM\n");
	//fprintf(modelfl,"multiclass_type=%d;\n", multiclass_type);
	fprintf(modelfl,"num_classes=%d;\n", m_labels->get_num_classes());
	fprintf(modelfl,"num_svms=%d;\n", m_machines.vlen);
	fprintf(modelfl,"kernel='%s';\n", m_kernel->get_name());

	for (int32_t i=0; i<m_machines.vlen; i++)
	{
		CSVM* svm=get_svm(i);
		ASSERT(svm);
		fprintf(modelfl,"\n%%SVM %d of %d\n", i, m_machines.vlen-1);
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

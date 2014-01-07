/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <io/SGIO.h>
#include <multiclass/MulticlassSVM.h>
#include <multiclass/MulticlassOneVsRestStrategy.h>

using namespace shogun;

CMulticlassSVM::CMulticlassSVM()
	:CKernelMulticlassMachine(new CMulticlassOneVsRestStrategy(), NULL, new CSVM(0), NULL)
{
	init();
}

CMulticlassSVM::CMulticlassSVM(CMulticlassStrategy *strategy)
	:CKernelMulticlassMachine(strategy, NULL, new CSVM(0), NULL)
{
	init();
}

CMulticlassSVM::CMulticlassSVM(
	CMulticlassStrategy *strategy, float64_t C, CKernel* k, CLabels* lab)
	: CKernelMulticlassMachine(strategy, k, new CSVM(C, k, lab), lab)
{
	init();
	m_C=C;
}

CMulticlassSVM::~CMulticlassSVM()
{
}

void CMulticlassSVM::init()
{
	SG_ADD(&m_C, "C", "C regularization constant",MS_AVAILABLE);
	m_C=0;
}

bool CMulticlassSVM::create_multiclass_svm(int32_t num_classes)
{
	if (num_classes>0)
	{
		int32_t num_svms=m_multiclass_strategy->get_num_machines();

		m_machines->reset_array();
		for (index_t i=0; i<num_svms; ++i)
			m_machines->push_back(NULL);

		return true;
	}
	return false;
}

bool CMulticlassSVM::set_svm(int32_t num, CSVM* svm)
{
	if (m_machines->get_num_elements()>0 && m_machines->get_num_elements()>num && num>=0 && svm)
	{
		m_machines->set_element(svm, num);
		return true;
	}
	return false;
}

bool CMulticlassSVM::init_machines_for_apply(CFeatures* data)
{
	if (is_data_locked())
	{
		SG_ERROR("CKernelMachine::apply(CFeatures*) cannot be called when "
				"data_lock was called before. Call data_unlock to allow.");
	}

	if (!m_kernel)
		SG_ERROR("No kernel assigned!\n")

	CFeatures* lhs=m_kernel->get_lhs();
	if (!lhs && m_kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("%s: No left hand side specified\n", get_name())

	if (m_kernel->get_kernel_type()!=K_COMBINED && !lhs->get_num_vectors())
	{
		SG_ERROR("%s: No vectors on left hand side (%s). This is probably due to"
				" an implementation error in %s, where it was forgotten to set "
				"the data (m_svs) indices\n", get_name(),
				data->get_name());
	}

	if (data && m_kernel->get_kernel_type()!=K_COMBINED)
		m_kernel->init(lhs, data);
	SG_UNREF(lhs);

	for (int32_t i=0; i<m_machines->get_num_elements(); i++)
	{
		CSVM *the_svm = (CSVM *)m_machines->get_element(i);
		ASSERT(the_svm)
		the_svm->set_kernel(m_kernel);
		SG_UNREF(the_svm);
	}

	return true;
}

bool CMulticlassSVM::load(FILE* modelfl)
{
	bool result=true;
	char char_buffer[1024];
	int32_t int_buffer;
	float64_t double_buffer;
	int32_t line_number=1;
	int32_t svm_idx=-1;

	SG_SET_LOCALE_C;

	if (fscanf(modelfl,"%15s\n", char_buffer)==EOF)
		SG_ERROR("error in svm file, line nr:%d\n", line_number)
	else
	{
		char_buffer[15]='\0';
		if (strcmp("%MultiClassSVM", char_buffer)!=0)
			SG_ERROR("error in multiclass svm file, line nr:%d\n", line_number)

		line_number++;
	}

	int_buffer=0;
	if (fscanf(modelfl," num_classes=%d; \n", &int_buffer) != 1)
		SG_ERROR("error in svm file, line nr:%d\n", line_number)

	if (!feof(modelfl))
		line_number++;

	if (int_buffer < 2)
		SG_ERROR("less than 2 classes - how is this multiclass?\n")

	create_multiclass_svm(int_buffer);

	int_buffer=0;
	if (fscanf(modelfl," num_svms=%d; \n", &int_buffer) != 1)
		SG_ERROR("error in svm file, line nr:%d\n", line_number)

	if (!feof(modelfl))
		line_number++;

	if (m_machines->get_num_elements() != int_buffer)
		SG_ERROR("Mismatch in number of svms: m_num_svms=%d vs m_num_svms(file)=%d\n", m_machines->get_num_elements(), int_buffer)

	if (fscanf(modelfl," kernel='%s'; \n", char_buffer) != 1)
		SG_ERROR("error in svm file, line nr:%d\n", line_number)

	if (!feof(modelfl))
		line_number++;

	for (int32_t n=0; n<m_machines->get_num_elements(); n++)
	{
		svm_idx=-1;
		if (fscanf(modelfl,"\n%4s %d of %d\n", char_buffer, &svm_idx, &int_buffer)==EOF)
		{
			result=false;
			SG_ERROR("error in svm file, line nr:%d\n", line_number)
		}
		else
		{
			char_buffer[4]='\0';
			if (strncmp("%SVM", char_buffer, 4)!=0)
			{
				result=false;
				SG_ERROR("error in svm file, line nr:%d\n", line_number)
			}

			if (svm_idx != n)
				SG_ERROR("svm index mismatch n=%d, n(file)=%d\n", n, svm_idx)

			line_number++;
		}

		int_buffer=0;
		if (fscanf(modelfl,"numsv%d=%d;\n", &svm_idx, &int_buffer) != 2)
			SG_ERROR("error in svm file, line nr:%d\n", line_number)

		if (svm_idx != n)
			SG_ERROR("svm index mismatch n=%d, n(file)=%d\n", n, svm_idx)

		if (!feof(modelfl))
			line_number++;

		SG_INFO("loading %ld support vectors for svm %d\n",int_buffer, svm_idx)
		CSVM* svm=new CSVM(int_buffer);

		double_buffer=0;

		if (fscanf(modelfl," b%d=%lf; \n", &svm_idx, &double_buffer) != 2)
			SG_ERROR("error in svm file, line nr:%d\n", line_number)

		if (svm_idx != n)
			SG_ERROR("svm index mismatch n=%d, n(file)=%d\n", n, svm_idx)

		if (!feof(modelfl))
			line_number++;

		svm->set_bias(double_buffer);

		if (fscanf(modelfl,"alphas%d=[\n", &svm_idx) != 1)
			SG_ERROR("error in svm file, line nr:%d\n", line_number)

		if (svm_idx != n)
			SG_ERROR("svm index mismatch n=%d, n(file)=%d\n", n, svm_idx)

		if (!feof(modelfl))
			line_number++;

		for (int32_t i=0; i<svm->get_num_support_vectors(); i++)
		{
			double_buffer=0;
			int_buffer=0;

			if (fscanf(modelfl,"\t[%lf,%d]; \n", &double_buffer, &int_buffer) != 2)
				SG_ERROR("error in svm file, line nr:%d\n", line_number)

			if (!feof(modelfl))
				line_number++;

			svm->set_support_vector(i, int_buffer);
			svm->set_alpha(i, double_buffer);
		}

		if (fscanf(modelfl,"%2s", char_buffer) == EOF)
		{
			result=false;
			SG_ERROR("error in svm file, line nr:%d\n", line_number)
		}
		else
		{
			char_buffer[3]='\0';
			if (strcmp("];", char_buffer)!=0)
			{
				result=false;
				SG_ERROR("error in svm file, line nr:%d\n", line_number)
			}
			line_number++;
		}

		set_svm(n, svm);
	}

	svm_proto()->svm_loaded=result;

	SG_RESET_LOCALE;
	return result;
}

bool CMulticlassSVM::save(FILE* modelfl)
{
	SG_SET_LOCALE_C;

	if (!m_kernel)
		SG_ERROR("Kernel not defined!\n")

	if (m_machines->get_num_elements()<1)
		SG_ERROR("Multiclass SVM not trained!\n")

	SG_INFO("Writing model file...")
	fprintf(modelfl,"%%MultiClassSVM\n");
	fprintf(modelfl,"num_classes=%d;\n", m_multiclass_strategy->get_num_classes());
	fprintf(modelfl,"num_svms=%d;\n", m_machines->get_num_elements());
	fprintf(modelfl,"kernel='%s';\n", m_kernel->get_name());

	for (int32_t i=0; i<m_machines->get_num_elements(); i++)
	{
		CSVM* svm=get_svm(i);
		ASSERT(svm)
		fprintf(modelfl,"\n%%SVM %d of %d\n", i, m_machines->get_num_elements()-1);
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
	SG_DONE()
	return true ;
}

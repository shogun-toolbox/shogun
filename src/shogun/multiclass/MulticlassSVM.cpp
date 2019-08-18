/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/multiclass/MulticlassSVM.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>

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
	SG_ADD(&m_C, "C", "C regularization constant",ParameterProperties::HYPER);
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
	if (!m_kernel)
		error("No kernel assigned!");

	CFeatures* lhs=m_kernel->get_lhs();
	if (!lhs && m_kernel->get_kernel_type()!=K_COMBINED)
		error("{}: No left hand side specified", get_name());

	if (m_kernel->get_kernel_type()!=K_COMBINED && !lhs->get_num_vectors())
	{
		error("{}: No vectors on left hand side ({}). This is probably due to"
				" an implementation error in {}, where it was forgotten to set "
				"the data (m_svs) indices", get_name(),
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
		error("error in svm file, line nr:{}", line_number);
	else
	{
		char_buffer[15]='\0';
		if (strcmp("%MultiClassSVM", char_buffer)!=0)
			error("error in multiclass svm file, line nr:{}", line_number);

		line_number++;
	}

	int_buffer=0;
	if (fscanf(modelfl," num_classes=%d; \n", &int_buffer) != 1)
		error("error in svm file, line nr:{}", line_number);

	if (!feof(modelfl))
		line_number++;

	if (int_buffer < 2)
		error("less than 2 classes - how is this multiclass?");

	create_multiclass_svm(int_buffer);

	int_buffer=0;
	if (fscanf(modelfl," num_svms=%d; \n", &int_buffer) != 1)
		error("error in svm file, line nr:{}", line_number);

	if (!feof(modelfl))
		line_number++;

	if (m_machines->get_num_elements() != int_buffer)
		error("Mismatch in number of svms: m_num_svms={} vs m_num_svms(file)={}", m_machines->get_num_elements(), int_buffer);

	if (fscanf(modelfl," kernel='%s'; \n", char_buffer) != 1)
		error("error in svm file, line nr:{}", line_number);

	if (!feof(modelfl))
		line_number++;

	for (int32_t n=0; n<m_machines->get_num_elements(); n++)
	{
		svm_idx=-1;
		if (fscanf(modelfl,"\n%4s %d of %d\n", char_buffer, &svm_idx, &int_buffer)==EOF)
		{
			result=false;
			error("error in svm file, line nr:{}", line_number);
		}
		else
		{
			char_buffer[4]='\0';
			if (strncmp("%SVM", char_buffer, 4)!=0)
			{
				result=false;
				error("error in svm file, line nr:{}", line_number);
			}

			if (svm_idx != n)
				error("svm index mismatch n={}, n(file)={}", n, svm_idx);

			line_number++;
		}

		int_buffer=0;
		if (fscanf(modelfl,"numsv%d=%d;\n", &svm_idx, &int_buffer) != 2)
			error("error in svm file, line nr:{}", line_number);

		if (svm_idx != n)
			error("svm index mismatch n={}, n(file)={}", n, svm_idx);

		if (!feof(modelfl))
			line_number++;

		io::info("loading {} support vectors for svm {}",int_buffer, svm_idx);
		CSVM* svm=new CSVM(int_buffer);

		double_buffer=0;

		if (fscanf(modelfl," b%d=%lf; \n", &svm_idx, &double_buffer) != 2)
			error("error in svm file, line nr:{}", line_number);

		if (svm_idx != n)
			error("svm index mismatch n={}, n(file)={}", n, svm_idx);

		if (!feof(modelfl))
			line_number++;

		svm->set_bias(double_buffer);

		if (fscanf(modelfl,"alphas%d=[\n", &svm_idx) != 1)
			error("error in svm file, line nr:{}", line_number);

		if (svm_idx != n)
			error("svm index mismatch n={}, n(file)={}", n, svm_idx);

		if (!feof(modelfl))
			line_number++;

		for (int32_t i=0; i<svm->get_num_support_vectors(); i++)
		{
			double_buffer=0;
			int_buffer=0;

			if (fscanf(modelfl,"\t[%lf,%d]; \n", &double_buffer, &int_buffer) != 2)
				error("error in svm file, line nr:{}", line_number);

			if (!feof(modelfl))
				line_number++;

			svm->set_support_vector(i, int_buffer);
			svm->set_alpha(i, double_buffer);
		}

		if (fscanf(modelfl,"%2s", char_buffer) == EOF)
		{
			result=false;
			error("error in svm file, line nr:{}", line_number);
		}
		else
		{
			char_buffer[3]='\0';
			if (strcmp("];", char_buffer)!=0)
			{
				result=false;
				error("error in svm file, line nr:{}", line_number);
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
		error("Kernel not defined!");

	if (m_machines->get_num_elements()<1)
		error("Multiclass SVM not trained!");

	io::info("Writing model file...");
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
	io::progress_done();
	return true ;
}

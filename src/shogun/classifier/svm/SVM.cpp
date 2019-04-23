/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Weijie Lin,
 *          Evgeniy Andreev, Viktor Gal, Evan Shelhamer, Thoralf Klein
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/Parameter.h>

#include <shogun/classifier/svm/SVM.h>
#include <shogun/classifier/mkl/MKL.h>
#include <shogun/labels/BinaryLabels.h>

#include <string.h>

using namespace shogun;

SVM::SVM(int32_t num_sv)
: KernelMachine()
{
	set_defaults(num_sv);
}

SVM::SVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab)
: KernelMachine()
{
	set_defaults();
	set_C(C,C);
	set_labels(lab);
	set_kernel(k);
}

SVM::~SVM()
{

}

void SVM::set_defaults(int32_t num_sv)
{
	SG_ADD(&C1, "C1", "", ParameterProperties::HYPER);
	SG_ADD(&C2, "C2", "", ParameterProperties::HYPER);
	SG_ADD(&svm_loaded, "svm_loaded", "SVM is loaded.");
	SG_ADD(&epsilon, "epsilon", "", ParameterProperties::HYPER);
	SG_ADD(&tube_epsilon, "tube_epsilon",
			"Tube epsilon for support vector regression.", ParameterProperties::HYPER);
	SG_ADD(&nu, "nu", "", ParameterProperties::HYPER);
	SG_ADD(&objective, "objective", "", ParameterProperties::HYPER);
	SG_ADD(&qpsize, "qpsize", "", ParameterProperties::HYPER);
	SG_ADD(&use_shrinking, "use_shrinking", "Shrinking shall be used.", ParameterProperties::SETTING);
	SG_ADD((std::shared_ptr<SGObject>*) &mkl, "mkl", "MKL object that svm optimizers need.");
	SG_ADD(&m_linear_term, "linear_term", "Linear term in qp.", ParameterProperties::MODEL);

	callback=NULL;
	mkl=NULL;

	set_loaded_status(false);

	set_epsilon(1e-5);
	set_tube_epsilon(1e-2);

	set_nu(0.5);
	set_C(1,1);

	set_objective(0);

	set_qpsize(41);
	set_bias_enabled(true);
	set_linadd_enabled(true);
	set_shrinking_enabled(true);
	set_batch_computation_enabled(true);

	if (num_sv>0)
		create_new_model(num_sv);
}

bool SVM::load(FILE* modelfl)
{
	bool result=true;
	char char_buffer[1024];
	int32_t int_buffer;
	float64_t double_buffer;
	int32_t line_number=1;

	SG_SET_LOCALE_C;

	if (fscanf(modelfl,"%4s\n", char_buffer)==EOF)
	{
		result=false;
		error("error in svm file, line nr:{}", line_number);
	}
	else
	{
		char_buffer[4]='\0';
		if (strcmp("%SVM", char_buffer)!=0)
		{
			result=false;
			error("error in svm file, line nr:{}", line_number);
		}
		line_number++;
	}

	int_buffer=0;
	if (fscanf(modelfl," numsv=%d; \n", &int_buffer) != 1)
	{
		result=false;
		error("error in svm file, line nr:{}", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	io::info("loading {} support vectors",int_buffer);
	create_new_model(int_buffer);

	if (fscanf(modelfl," kernel='%s'; \n", char_buffer) != 1)
	{
		result=false;
		error("error in svm file, line nr:{}", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	double_buffer=0;

	if (fscanf(modelfl," b=%lf; \n", &double_buffer) != 1)
	{
		result=false;
		error("error in svm file, line nr:{}", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	set_bias(double_buffer);

	if (fscanf(modelfl,"%8s\n", char_buffer) == EOF)
	{
		result=false;
		error("error in svm file, line nr:{}", line_number);
	}
	else
	{
		char_buffer[9]='\0';
		if (strcmp("alphas=[", char_buffer)!=0)
		{
			result=false;
			error("error in svm file, line nr:{}", line_number);
		}
		line_number++;
	}

	for (int32_t i=0; i<get_num_support_vectors(); i++)
	{
		double_buffer=0;
		int_buffer=0;

		if (fscanf(modelfl," \[%lf,%d]; \n", &double_buffer, &int_buffer) != 2)
		{
			result=false;
			error("error in svm file, line nr:{}", line_number);
		}

		if (!feof(modelfl))
			line_number++;

		set_support_vector(i, int_buffer);
		set_alpha(i, double_buffer);
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

	set_loaded_status(result);
	SG_RESET_LOCALE;
	return result;
}

bool SVM::save(FILE* modelfl)
{
	SG_SET_LOCALE_C;

	if (!kernel)
		error("Kernel not defined!");

	io::info("Writing model file...");
	fprintf(modelfl,"%%SVM\n");
	fprintf(modelfl,"numsv=%d;\n", get_num_support_vectors());
	fprintf(modelfl,"kernel='%s';\n", kernel->get_name());
	fprintf(modelfl,"b=%+10.16e;\n",get_bias());

	fprintf(modelfl, "alphas=\[\n");

	for(int32_t i=0; i<get_num_support_vectors(); i++)
		fprintf(modelfl,"\t[%+10.16e,%d];\n",
				SVM::get_alpha(i), get_support_vector(i));

	fprintf(modelfl, "];\n");

	io::progress_done();
	SG_RESET_LOCALE;
	return true ;
}

void SVM::set_callback_function(std::shared_ptr<MKL> m, bool (*cb)
		(std::shared_ptr<MKL> mkl, const float64_t* sumw, const float64_t suma))
{


	mkl=m;

	callback=cb;
}

float64_t SVM::compute_svm_dual_objective()
{
	int32_t n=get_num_support_vectors();

	if (m_labels && kernel)
	{
		auto binary_labels = std::static_pointer_cast<BinaryLabels>(m_labels);
		objective=0;
		for (int32_t i=0; i<n; i++)
		{
			int32_t ii=get_support_vector(i);
			objective-=get_alpha(i)*(binary_labels->get_label(ii));

			for (int32_t j=0; j<n; j++)
			{
				int32_t jj=get_support_vector(j);
				objective+=0.5*get_alpha(i)*get_alpha(j)*kernel->kernel(ii,jj);
			}
		}
	}
	else
		error("cannot compute objective, labels or kernel not set");

	return objective;
}

float64_t SVM::compute_svm_primal_objective()
{
	int32_t n=get_num_support_vectors();
	float64_t regularizer=0;
	float64_t loss=0;



	if (m_labels && kernel)
	{
		float64_t C2_tmp=get_C1();
		auto binary_labels = std::static_pointer_cast<BinaryLabels>(m_labels);
		if(C2>0)
		{
			C2_tmp=get_C2();
		}

		for (int32_t i=0; i<n; i++)
		{
			int32_t ii=get_support_vector(i);
			for (int32_t j=0; j<n; j++)
			{
				int32_t jj=get_support_vector(j);
				regularizer-=0.5*get_alpha(i)*get_alpha(j)*kernel->kernel(ii,jj);
			}

			loss-=(C1*(-(binary_labels->get_label(ii)+1)/2.0 + C2_tmp*(binary_labels->get_label(ii)+1)/2.0 )*Math::max(0.0, 1.0-binary_labels->get_label(ii)*apply_one(ii)));
		}

	}
	else
		error("cannot compute objective, labels or kernel not set");

	return regularizer+loss;
}

float64_t* SVM::get_linear_term_array()
{
	if (m_linear_term.vlen==0)
		return NULL;
	float64_t* a = SG_MALLOC(float64_t, m_linear_term.vlen);

	sg_memcpy(a, m_linear_term.vector,
			m_linear_term.vlen*sizeof(float64_t));

	return a;
}

void SVM::set_linear_term(const SGVector<float64_t>& linear_term)
{
	ASSERT(linear_term.vector)

	if (!m_labels)
		error("Please assign labels first!");

	int32_t num_labels=m_labels->get_num_labels();

	if (num_labels != linear_term.vlen)
	{
		error("Number of labels ({}) does not match number"
				"of entries ({}) in linear term \n", num_labels, linear_term.vlen);
	}

	m_linear_term=linear_term;
}

SGVector<float64_t> SVM::get_linear_term()
{
	return m_linear_term;
}

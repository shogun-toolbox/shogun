#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "lib/io.h"

#include <string.h>

CSVM::CSVM()
{
	CKernelMachine::kernel=NULL;

	svm_model.b=0.0;
	svm_model.alpha=NULL;
	svm_model.svs=NULL;
	svm_model.num_svs=0;

	svm_loaded=false;

	qpsize=41;
	weight_epsilon=1e-5;
	C1=1;
	C2=1;
	C_mkl=0;
	weight_epsilon=1e-5;
	epsilon=1e-5;
	use_mkl = false;
	use_linadd = false;
	use_precomputed_subkernels = false ;
}

CSVM::~CSVM()
{
  delete[] svm_model.alpha ;
  delete[] svm_model.svs ;

  CIO::message(M_DEBUG, "SVM object destroyed\n") ;
}

bool CSVM::load(FILE* modelfl)
{
	bool result=true;
	CHAR char_buffer[1024];
	int int_buffer;
	double double_buffer;
	int line_number=1;

	if (fscanf(modelfl,"%4s\n", char_buffer)==EOF)
	{
		result=false;
		CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
	}
	else
	{
		char_buffer[4]='\0';
		if (strcmp("%SVM", char_buffer)!=0)
		{
			result=false;
			CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
		}
		line_number++;
	}

	int_buffer=0;
	if (fscanf(modelfl," numsv=%d; \n", &int_buffer) != 1)
	{
		result=false;
		CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	CIO::message(M_INFO, "loading %ld support vectors\n",int_buffer);
	create_new_model(int_buffer);

	if (fscanf(modelfl," kernel='%s'; \n", char_buffer) != 1)
	{
		result=false;
		CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	double_buffer=0;
	
	if (fscanf(modelfl," b=%lf; \n", &double_buffer) != 1)
	{
		result=false;
		CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
	}
	
	if (!feof(modelfl))
		line_number++;

	set_bias(double_buffer);

	if (fscanf(modelfl,"%8s\n", char_buffer) == EOF)
	{
		result=false;
		CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
	}
	else
	{
		char_buffer[9]='\0';
		if (strcmp("alphas=[", char_buffer)!=0)
		{
			result=false;
			CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
		}
		line_number++;
	}

	for (INT i=0; i<get_num_support_vectors(); i++)
	{
		double_buffer=0;
		int_buffer=0;

		if (fscanf(modelfl," \[%lf,%d]; \n", &double_buffer, &int_buffer) != 2)
		{
			result=false;
			CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
		}

		if (!feof(modelfl))
			line_number++;

		set_support_vector(i, int_buffer);
		set_alpha(i, double_buffer);
	}

	if (fscanf(modelfl,"%2s", char_buffer) == EOF)
	{
		result=false;
		CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
	}
	else
	{
		char_buffer[3]='\0';
		if (strcmp("];", char_buffer)!=0)
		{
			result=false;
			CIO::message(M_ERROR, "error in svm file, line nr:%d\n", line_number);
		}
		line_number++;
	}

	svm_loaded=result;
	return result;
}

bool CSVM::save(FILE* modelfl)
{
  CIO::message(M_INFO, "Writing model file...");
  fprintf(modelfl,"%%SVM\n");
  fprintf(modelfl,"numsv=%d;\n", get_num_support_vectors());
  fprintf(modelfl,"kernel='%s';\n", CKernelMachine::get_kernel()->get_name());
  fprintf(modelfl,"b=%+10.16e;\n",get_bias());

  fprintf(modelfl, "alphas=\[\n");
  
  for(INT i=0; i<get_num_support_vectors(); i++)
    fprintf(modelfl,"\t[%+10.16e,%d];\n", CSVM::get_alpha(i), get_support_vector(i));

  fprintf(modelfl, "];\n");
  
  CIO::message(M_INFO, "done\n");
  return true ;
} 

REAL* CSVM::test()
{
  if (!CKernelMachine::get_kernel())
  {
      CIO::message(M_ERROR, "SVM can not proceed without kernel!\n");
      return false ;
  }

  CLabels* lab=CKernelMachine::get_labels();
  assert(lab!=NULL);
  INT num_test=lab->get_num_labels();

  CIO::message(M_DEBUG, "%d test examples\n", num_test);
  REAL* output=new REAL[num_test];

  for (INT i=0; i<num_test;  i++)
  {
	  if ( (i% (num_test/100+1))== 0)
		  CIO::progress(i, 0, num_test-1);

	  output[i]=classify_example(i);
  }
  CIO::message(M_MESSAGEONLY, "done.           \n");
  return output;
}

CLabels* CSVM::classify(CLabels* result)
{
	if (!CKernelMachine::get_kernel())
	{
		CIO::message(M_ERROR, "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( (CKernelMachine::get_kernel()) &&
			(CKernelMachine::get_kernel())->get_rhs() &&
			(CKernelMachine::get_kernel())->get_rhs()->get_num_vectors())
	{
		INT num_vectors=(CKernelMachine::get_kernel())->get_rhs()->get_num_vectors();

		if (!result)
			result=new CLabels(num_vectors);

		assert(result);
		CIO::message(M_DEBUG, "computing output on %d test examples\n", num_vectors);

		CIO::message(M_DEBUG, "using optimized kernel\n");
		for (INT vec=0; vec<num_vectors; vec++)
		{
			if ( (vec% (num_vectors/10+1))== 0)
				CIO::progress(vec, 0, num_vectors-1);

			result->set_label(vec, classify_example(vec));
		}
		CIO::message(M_MESSAGEONLY, "done.           \n");
	}
	else 
		return NULL;

	return result;
}

REAL CSVM::classify_example(INT num)
{
	if (CKernelMachine::get_kernel() && CKernelMachine::get_kernel()->is_optimizable() && (CKernelMachine::get_kernel()->get_is_initialized()))
	{
		REAL dist = CKernelMachine::get_kernel()->compute_optimized(num);
		return (dist+get_bias());
	}
	else
	{
		REAL dist=0;
		for(INT i=0; i<get_num_support_vectors(); i++)
			dist+=CKernelMachine::get_kernel()->kernel(get_support_vector(i), num)*get_alpha(i);

		return (dist+get_bias());
	}
}

#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "lib/io.h"
#include "kernel/WeightedDegreeCharKernel.h"

#include <string.h>

CSVM::CSVM()
{
	CKernelMachine::kernel=NULL;

	svm_model.b=0.0;
	svm_model.alpha=NULL;
	svm_model.svs=NULL;
	svm_model.num_svs=0;

	svm_loaded=false;

	C1=-1;
	C2=-1;
}

CSVM::~CSVM()
{
  delete[] svm_model.alpha ;
  delete[] svm_model.svs ;

  CIO::message("SVM object destroyed\n") ;
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
		CIO::message("error in svm file, line nr:%d\n", line_number);
	}
	else
	{
		char_buffer[4]='\0';
		if (strcmp("%SVM", char_buffer)!=0)
		{
			result=false;
			CIO::message("error in svm file, line nr:%d\n", line_number);
		}
		line_number++;
	}

	int_buffer=0;
	if (fscanf(modelfl," numsv=%d; \n", &int_buffer) != 1)
	{
		result=false;
		CIO::message("error in svm file, line nr:%d\n", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	CIO::message("loading %ld support vectors\n",int_buffer);
	create_new_model(int_buffer);

	if (fscanf(modelfl," kernel='%s'; \n", char_buffer) != 1)
	{
		result=false;
		CIO::message("error in svm file, line nr:%d\n", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	double_buffer=0;
	
	if (fscanf(modelfl," b=%lf; \n", &double_buffer) != 1)
	{
		result=false;
		CIO::message("error in svm file, line nr:%d\n", line_number);
		CIO::message("b\n");
	}
	
	if (!feof(modelfl))
		line_number++;

	set_bias(double_buffer);

	if (fscanf(modelfl,"%8s\n", char_buffer) == EOF)
	{
		result=false;
		CIO::message("error in svm file, line nr:%d\n", line_number);
	}
	else
	{
		char_buffer[9]='\0';
		if (strcmp("alphas=[", char_buffer)!=0)
		{
			result=false;
			CIO::message("error in svm file, line nr:%d\n", line_number);
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
			CIO::message("error in svm file, line nr:%d\n", line_number);
		}

		if (!feof(modelfl))
			line_number++;

		set_support_vector(i, int_buffer);
		set_alpha(i, double_buffer);
	}

	if (fscanf(modelfl,"%2s", char_buffer) == EOF)
	{
		result=false;
		CIO::message("error in svm file, line nr:%d\n", line_number);
	}
	else
	{
		char_buffer[3]='\0';
		if (strcmp("];", char_buffer)!=0)
		{
			result=false;
			CIO::message("error in svm file, line nr:%d\n", line_number);
		}
		line_number++;
	}

	svm_loaded=result;
	return result;
}

bool CSVM::save(FILE* modelfl)
{
  CIO::message("Writing model file...");
  fprintf(modelfl,"%%SVM\n");
  fprintf(modelfl,"numsv=%d;\n", get_num_support_vectors());
  fprintf(modelfl,"kernel='%s';\n", CKernelMachine::get_kernel()->get_name());
  fprintf(modelfl,"b=%+10.16e;\n",get_bias());

  fprintf(modelfl, "alphas=\[\n");
  
  for(INT i=0; i<get_num_support_vectors(); i++)
    fprintf(modelfl,"\t[%+10.16e,%d];\n", CSVM::get_alpha(i), get_support_vector(i));

  fprintf(modelfl, "];\n");
  
  CIO::message("done\n");
  return true ;
} 

REAL* CSVM::test()
{
  if (!CKernelMachine::get_kernel())
  {
      CIO::message("SVM can not proceed without kernel!\n");
      return false ;
  }

  CLabels* lab=CKernelMachine::get_labels();
  assert(lab!=NULL);
  INT num_test=lab->get_num_labels();

  CIO::message("%d test examples\n", num_test);
  REAL* output=new REAL[num_test];

  for (INT i=0; i<num_test;  i++)
  {
	  if ( (i% (num_test/10+1))== 0)
		  CIO::message("%i%%..",100*i/(num_test+1));

	  output[i]=classify_example(i);
  }
  CIO::message(".done.\n");
  return output;
}

CLabels* CSVM::classify(CLabels* result)
{
	if (!CKernelMachine::get_kernel())
	{
		CIO::message("SVM can not proceed without kernel!\n");
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
		CIO::message("num vec: %d\n", num_vectors);

		CWeightedDegreeCharKernel *kernel=
			(CWeightedDegreeCharKernel*)CKernelMachine::get_kernel() ;

		if ((kernel->get_kernel_type() == K_WEIGHTEDDEGREE) && 
			(kernel->is_tree_initialized()))
		{
			for (INT vec=0; vec<num_vectors; vec++)
			{
				REAL res=classify_example_wd(vec) ;
				result->set_label(vec, res);
			}
		}
		else
		{
			for (INT vec=0; vec<num_vectors; vec++)
				result->set_label(vec, classify_example(vec));
		}
	}
	else 
		return NULL;

	return result;
}

REAL CSVM::classify_example(INT num)
{
	REAL dist=0;
	for(INT i=0; i<get_num_support_vectors(); i++)
	{
		CIO::message("i:%d\n\tsv_idx:%d\n ", i, get_support_vector(i));
		CIO::message("\talpha:%f\n ", get_alpha(i));
		CIO::message("\tnum_sv:%d num:%d\n", get_num_support_vectors(), num);
		dist+=CKernelMachine::get_kernel()->kernel(get_support_vector(i), num)*get_alpha(i);
	}
	
	return(dist+get_bias());
}

REAL CSVM::classify_example_wd(INT num)
{
	CWeightedDegreeCharKernel *kernel=
		(CWeightedDegreeCharKernel*)CKernelMachine::get_kernel() ;
	
	REAL dist=kernel->compute_by_tree(num) ;

	return(dist+get_bias());
}

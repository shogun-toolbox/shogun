#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "lib/io.h"

#include <string.h>

CSVM::CSVM()
{
	C=-1;
	CKernelMachine::kernel=NULL;
	svm_loaded=false;
	svm_model.b=0.0;
	svm_model.alpha=NULL;
	svm_model.svs=NULL;
	svm_model.num_svs=0;
}

CSVM::~CSVM()
{
  delete[] svm_model.alpha ;
  delete[] svm_model.svs ;

  CIO::message("SVM object destroyed\n") ;
}

bool CSVM::load(FILE* modelfl)
{
	bool result=false;
	CHAR char_buffer[1024];
	int int_buffer;
	double double_buffer;

	memset(&svm_model, 0x0, sizeof(TModel));

	fscanf(modelfl,"%%SVM\n");
	fscanf(modelfl,"numsv=%d%*[^\n]\n", &int_buffer);
	svm_model.num_svs=int_buffer;
	fscanf(modelfl,"kernel='%s'\n", char_buffer);
	fscanf(modelfl,"b=%lf%*[^\n]\n", &double_buffer);
	svm_model.b=double_buffer;

	CIO::message("loading %ld support vectors\n",svm_model.num_svs);
	create_new_model(svm_model.num_svs);

	fscanf(modelfl,"alphas=\[\n");

	for (INT i=1; i<svm_model.num_svs; i++)
	{
		fscanf(modelfl,"\t[%lf,%d];%*[^\n]\n", &double_buffer, &int_buffer);
		set_support_vector(i, int_buffer);
		set_alpha(i, double_buffer);
	}

	fscanf(modelfl,"];");

	result=true;
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
  
  for(INT i=0; i<svm_model.num_svs; i++)
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

		for (INT vec=0; vec<num_vectors; vec++)
			result->set_label(vec, classify_example(vec));
	}
	else 
		return NULL;

	return result;
}

REAL CSVM::classify_example(INT num)
{
  REAL dist=0;
  for(INT i=0; i<get_num_support_vectors(); i++)
    dist+=CKernelMachine::get_kernel()->kernel(get_support_vector(i), num)*get_alpha(i);
  
	//CIO::message("%f %f abused\n", dist, dist+get_bias());
  return(dist+get_bias());
}

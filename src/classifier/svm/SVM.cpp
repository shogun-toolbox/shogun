/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Signal.h"

#include "classifier/svm/SVM.h"

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
	nu=0.5;
	C1=1;
	C2=1;
	C_mkl=0;
	weight_epsilon=1e-5;
	epsilon=1e-5;
	use_mkl = false;
	use_batch_computation = true;
	use_shrinking= true;
	use_linadd = false;
	use_precomputed_subkernels = false ;
	objective=0;
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

bool CSVM::init_kernel_optimization()
{
	if (get_kernel() && get_kernel()->has_property(KP_LINADD) && get_num_support_vectors())
	{
		INT * sv_idx    = new INT[get_num_support_vectors()] ;
		DREAL* sv_weight = new DREAL[get_num_support_vectors()] ;

		for(INT i=0; i<get_num_support_vectors(); i++)
		{
			sv_idx[i]    = get_support_vector(i) ;
			sv_weight[i] = get_alpha(i) ;
		}

		bool ret = kernel->init_optimization(get_num_support_vectors(), sv_idx, sv_weight) ;

		delete[] sv_idx ;
		delete[] sv_weight ;

		if (!ret)
			CIO::message(M_ERROR, "initialization of kernel optimization failed\n") ;

		return ret;
	}

	return false;
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

		ASSERT(result);
		CIO::message(M_DEBUG, "computing output on %d test examples\n", num_vectors);

		if (CKernelMachine::get_kernel()->has_property(KP_BATCHEVALUATION) && get_batch_computation_enabled())
		{
			INT num_vec=0;
			ASSERT(get_num_support_vectors()>0);
			INT * sv_idx    = new INT[get_num_support_vectors()] ;
			DREAL* sv_weight = new DREAL[get_num_support_vectors()] ;

			for (INT i=0; i<get_num_support_vectors(); i++)
			{
				sv_idx[i]    = get_support_vector(i) ;
				sv_weight[i] = get_alpha(i) ;
			}

			DREAL* r=CKernelMachine::get_kernel()->compute_batch(num_vec, NULL, get_num_support_vectors(), sv_idx, sv_weight);
			ASSERT(num_vec==num_vectors);

			for (INT i=0; i<num_vec && !CSignal::cancel_computations(); i++)
				result->set_label(i, get_bias()+r[i]);

			delete[] sv_idx ;
			delete[] sv_weight ;
			delete[] r;
		}
		else
		{
			for (INT vec=0; vec<num_vectors && !CSignal::cancel_computations(); vec++)
			{
				if ( (vec% (num_vectors/10+1))== 0)
					CIO::progress(vec, 0.0, num_vectors-1);

				result->set_label(vec, classify_example(vec));
			}
		}
		if ( CSignal::cancel_computations() )
			CIO::message(M_INFO, "prematurely stopped.           \n");
		else
			CIO::message(M_INFO, "done.           \n");
	}
	else 
		return NULL;

	return result;
}

DREAL CSVM::classify_example(INT num)
{
	ASSERT(CKernelMachine::get_kernel());

	if (CKernelMachine::get_kernel()->has_property(KP_LINADD) && (CKernelMachine::get_kernel()->get_is_initialized()))
	{
		DREAL dist = CKernelMachine::get_kernel()->compute_optimized(num);
		return (dist+get_bias());
	}
	else
	{
		DREAL dist=0;
		for(INT i=0; i<get_num_support_vectors(); i++)
			dist+=CKernelMachine::get_kernel()->kernel(get_support_vector(i), num)*get_alpha(i);

		return (dist+get_bias());
	}
}


DREAL CSVM::compute_objective()
{
	CLabels* lab=CKernelMachine::get_labels();
	INT n=get_num_support_vectors();
	CKernel* k=CKernelMachine::get_kernel();

	if (lab && k)
	{
		ASSERT(lab);
		ASSERT(k);

		objective=0;
		for (int i=0; i<n; i++)
		{
			objective-=get_alpha(i)*lab->get_label(i);
			for (int j=0; j<n; j++)
				objective+=0.5*get_alpha(i)*get_alpha(j)*k->kernel(i,j);
		}
	}
	else
		CIO::message(M_ERROR, "cannot compute objective, labels or kernel not set\n");

	return objective;
}

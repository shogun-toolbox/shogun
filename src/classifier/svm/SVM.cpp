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
#include "lib/Parallel.h"

#include "classifier/svm/SVM.h"

#include <string.h>
#include <pthread.h>

struct S_THREAD_PARAM 
{
	CSVM* svm;
	CLabels* result;
	INT start;
	INT end;
	bool verbose;
};

CSVM::CSVM(INT num_sv)
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
	use_linadd = true;
	use_precomputed_subkernels = false ;
	objective=0;

    if (num_sv>0)
        create_new_model(num_sv);
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
	INT num_sv=get_num_support_vectors();

	if (get_kernel() && get_kernel()->has_property(KP_LINADD) && num_sv>0)
	{
		INT * sv_idx    = new INT[num_sv] ;
		DREAL* sv_weight = new DREAL[num_sv] ;

		for(INT i=0; i<num_sv; i++)
		{
			sv_idx[i]    = get_support_vector(i) ;
			sv_weight[i] = get_alpha(i) ;
		}

		bool ret = kernel->init_optimization(num_sv, sv_idx, sv_weight) ;

		delete[] sv_idx ;
		delete[] sv_weight ;

		if (!ret)
			CIO::message(M_ERROR, "initialization of kernel optimization failed\n") ;

		return ret;
	}
	else
		CIO::message(M_ERROR, "initialization of kernel optimization failed\n") ;

	return false;
}

void* CSVM::classify_example_helper(void* p)
{
	S_THREAD_PARAM* params= (S_THREAD_PARAM*) p;
	CLabels* result=params->result;
	CSVM* svm=params->svm;

#ifdef CYGWIN
	for (INT vec=params->start; vec<params->end; vec++)
#else
	for (INT vec=params->start; vec<params->end && !CSignal::cancel_computations(); vec++)
#endif
	{
		if (params->verbose)
		{
			INT num_vectors=params->end - params->start;
			INT v=vec-params->start;
			if ( (v% (num_vectors/100+1))== 0)
				CIO::progress(v, 0.0, num_vectors-1);
		}

		result->set_label(vec, svm->classify_example(vec));
	}

	return NULL;
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
			ASSERT(get_num_support_vectors()>0);
			INT * sv_idx    = new INT[get_num_support_vectors()] ;
			DREAL* sv_weight = new DREAL[get_num_support_vectors()] ;
			INT* idx = new INT[num_vectors];
			DREAL* output = new DREAL[num_vectors];

			ASSERT(sv_idx);
			ASSERT(sv_weight);

			ASSERT(idx);
			ASSERT(output);
			memset(output, 0, sizeof(DREAL)*num_vectors);

			//compute output for all vectors v[0]...v[num_vectors-1]
			for (INT i=0; i<num_vectors; i++)
				idx[i]=i;

			for (INT i=0; i<get_num_support_vectors(); i++)
			{
				sv_idx[i]    = get_support_vector(i) ;
				sv_weight[i] = get_alpha(i) ;
			}

			CKernelMachine::get_kernel()->compute_batch(num_vectors, idx, output, get_num_support_vectors(), sv_idx, sv_weight);

			for (INT i=0; i<num_vectors; i++)
				result->set_label(i, get_bias()+output[i]);

			delete[] sv_idx ;
			delete[] sv_weight ;
			delete[] idx;
			delete[] output;
		}
		else
		{
			INT num_threads=CParallel::get_num_threads();
			ASSERT(num_threads>0);

			if (num_threads < 2)
			{
				S_THREAD_PARAM params;
				params.svm=this;
				params.result=result;
				params.start=0;
				params.end=num_vectors;
				params.verbose=true;
				classify_example_helper((void*) &params);
			}
			else
			{
				pthread_t threads[num_threads-1];
				S_THREAD_PARAM params[num_threads];
				INT step= num_vectors/num_threads;

				INT t;

				for (t=0; t<num_threads-1; t++)
				{
					params[t].svm = this;
					params[t].result = result;
					params[t].start = t*step;
					params[t].end = (t+1)*step;
					params[t].verbose = false;
					pthread_create(&threads[t], NULL, CSVM::classify_example_helper, (void*)&params[t]);
				}

				params[t].svm = this;
				params[t].result = result;
				params[t].start = t*step;
				params[t].end = num_vectors;
				params[t].verbose = true;
				classify_example_helper((void*) &params[t]);

				for (t=0; t<num_threads-1; t++)
					pthread_join(threads[t], NULL);
			}
		}

#ifndef CYGWIN
		if ( CSignal::cancel_computations() )
			CIO::message(M_INFO, "prematurely stopped.           \n");
		else
#endif
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

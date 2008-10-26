/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Signal.h"
#include "base/Parallel.h"

#include "classifier/svm/SVM.h"

#include <string.h>

#ifndef WIN32
#include <pthread.h>
#endif

struct S_THREAD_PARAM
{
	CSVM* svm;
	CLabels* result;
	int32_t start;
	int32_t end;
	bool verbose;
};

CSVM::CSVM(int32_t num_sv)
: CKernelMachine()
{
	set_defaults(num_sv);
}

CSVM::CSVM(DREAL C, CKernel* k, CLabels* lab)
: CKernelMachine()
{
	set_defaults();
	set_C(C,C);
	set_labels(lab);
	set_kernel(k);
}

CSVM::~CSVM()
{
	delete[] svm_model.alpha;
	delete[] svm_model.svs;

	SG_DEBUG("SVM object destroyed\n");
}

void CSVM::set_defaults(int32_t num_sv)
{
	svm_model.b=0.0;
	svm_model.alpha=NULL;
	svm_model.svs=NULL;
	svm_model.num_svs=0;
	svm_loaded=false;

	weight_epsilon=1e-5;
	epsilon=1e-5;
	tube_epsilon=1e-2;

	nu=0.5;
	C1=1;
	C2=1;
	C_mkl=0;
	mkl_norm=1;

	objective=0;

	qpsize=41;
	use_bias=true;
	use_shrinking=true;
	use_mkl=false;
	use_batch_computation=true;
	use_linadd=true;

    if (num_sv>0)
        create_new_model(num_sv);
}

bool CSVM::load(FILE* modelfl)
{
	bool result=true;
	char char_buffer[1024];
	int32_t int_buffer;
	double double_buffer;
	int32_t line_number=1;

	if (fscanf(modelfl,"%4s\n", char_buffer)==EOF)
	{
		result=false;
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);
	}
	else
	{
		char_buffer[4]='\0';
		if (strcmp("%SVM", char_buffer)!=0)
		{
			result=false;
			SG_ERROR( "error in svm file, line nr:%d\n", line_number);
		}
		line_number++;
	}

	int_buffer=0;
	if (fscanf(modelfl," numsv=%d; \n", &int_buffer) != 1)
	{
		result=false;
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	SG_INFO( "loading %ld support vectors\n",int_buffer);
	create_new_model(int_buffer);

	if (fscanf(modelfl," kernel='%s'; \n", char_buffer) != 1)
	{
		result=false;
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);
	}

	if (!feof(modelfl))
		line_number++;

	double_buffer=0;
	
	if (fscanf(modelfl," b=%lf; \n", &double_buffer) != 1)
	{
		result=false;
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);
	}
	
	if (!feof(modelfl))
		line_number++;

	set_bias(double_buffer);

	if (fscanf(modelfl,"%8s\n", char_buffer) == EOF)
	{
		result=false;
		SG_ERROR( "error in svm file, line nr:%d\n", line_number);
	}
	else
	{
		char_buffer[9]='\0';
		if (strcmp("alphas=[", char_buffer)!=0)
		{
			result=false;
			SG_ERROR( "error in svm file, line nr:%d\n", line_number);
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
			SG_ERROR( "error in svm file, line nr:%d\n", line_number);
		}

		if (!feof(modelfl))
			line_number++;

		set_support_vector(i, int_buffer);
		set_alpha(i, double_buffer);
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

	svm_loaded=result;
	return result;
}

bool CSVM::save(FILE* modelfl)
{
	if (!kernel)
		SG_ERROR("Kernel not defined!\n");

	SG_INFO( "Writing model file...");
	fprintf(modelfl,"%%SVM\n");
	fprintf(modelfl,"numsv=%d;\n", get_num_support_vectors());
	fprintf(modelfl,"kernel='%s';\n", kernel->get_name());
	fprintf(modelfl,"b=%+10.16e;\n",get_bias());

	fprintf(modelfl, "alphas=\[\n");

	for(int32_t i=0; i<get_num_support_vectors(); i++)
		fprintf(modelfl,"\t[%+10.16e,%d];\n",
				CSVM::get_alpha(i), get_support_vector(i));

	fprintf(modelfl, "];\n");

	SG_DONE();
	return true ;
} 

bool CSVM::init_kernel_optimization()
{
	int32_t num_sv=get_num_support_vectors();

	if (kernel && kernel->has_property(KP_LINADD) && num_sv>0)
	{
		int32_t * sv_idx    = new int32_t[num_sv] ;
		DREAL* sv_weight = new DREAL[num_sv] ;

		for(int32_t i=0; i<num_sv; i++)
		{
			sv_idx[i]    = get_support_vector(i) ;
			sv_weight[i] = get_alpha(i) ;
		}

		bool ret = kernel->init_optimization(num_sv, sv_idx, sv_weight) ;

		delete[] sv_idx ;
		delete[] sv_weight ;

		if (!ret)
			SG_ERROR( "initialization of kernel optimization failed\n");

		return ret;
	}
	else
		SG_ERROR( "initialization of kernel optimization failed\n");

	return false;
}

void* CSVM::classify_example_helper(void* p)
{
	S_THREAD_PARAM* params= (S_THREAD_PARAM*) p;
	CLabels* result=params->result;
	CSVM* svm=params->svm;

#ifdef WIN32
	for (int32_t vec=params->start; vec<params->end; vec++)
#else
	CSignal::clear_cancel();
	for (int32_t vec=params->start; vec<params->end && 
			!CSignal::cancel_computations(); vec++)
#endif
	{
		if (params->verbose)
		{
			int32_t num_vectors=params->end - params->start;
			int32_t v=vec-params->start;
			if ( (v% (num_vectors/100+1))== 0)
				SG_SPROGRESS(v, 0.0, num_vectors-1);
		}

		result->set_label(vec, svm->classify_example(vec));
	}

	return NULL;
}

CLabels* CSVM::classify(CLabels* lab)
{
	if (!kernel)
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( kernel && kernel->get_num_vec_rhs()>0 )
	{
		int32_t num_vectors=kernel->get_num_vec_rhs();

		if (!lab)
			lab=new CLabels(num_vectors);
		SG_DEBUG( "computing output on %d test examples\n", num_vectors);

		if (this->io.get_show_progress())
			this->io.enable_progress();
		else
			this->io.disable_progress();

		if (kernel->has_property(KP_BATCHEVALUATION) &&
				get_batch_computation_enabled())
		{
			ASSERT(get_num_support_vectors()>0);
			int32_t* sv_idx=new int32_t[get_num_support_vectors()];
			DREAL* sv_weight=new DREAL[get_num_support_vectors()];
			int32_t* idx=new int32_t[num_vectors];
			DREAL* output=new DREAL[num_vectors];
			memset(output, 0, sizeof(DREAL)*num_vectors);

			//compute output for all vectors v[0]...v[num_vectors-1]
			for (int32_t i=0; i<num_vectors; i++)
				idx[i]=i;

			for (int32_t i=0; i<get_num_support_vectors(); i++)
			{
				sv_idx[i]    = get_support_vector(i) ;
				sv_weight[i] = get_alpha(i) ;
			}

			kernel->compute_batch(num_vectors, idx,
					output, get_num_support_vectors(), sv_idx, sv_weight);

			for (int32_t i=0; i<num_vectors; i++)
				lab->set_label(i, get_bias()+output[i]);

			delete[] sv_idx ;
			delete[] sv_weight ;
			delete[] idx;
			delete[] output;
		}
		else
		{
			int32_t num_threads=parallel.get_num_threads();
			ASSERT(num_threads>0);

			if (num_threads < 2)
			{
				S_THREAD_PARAM params;
				params.svm=this;
				params.result=lab;
				params.start=0;
				params.end=num_vectors;
				params.verbose=true;
				classify_example_helper((void*) &params);
			}
#ifndef WIN32
			else
			{
				pthread_t threads[num_threads-1];
				S_THREAD_PARAM params[num_threads];
				int32_t step= num_vectors/num_threads;

				int32_t t;

				for (t=0; t<num_threads-1; t++)
				{
					params[t].svm = this;
					params[t].result = lab;
					params[t].start = t*step;
					params[t].end = (t+1)*step;
					params[t].verbose = false;
					pthread_create(&threads[t], NULL,
							CSVM::classify_example_helper, (void*)&params[t]);
				}

				params[t].svm = this;
				params[t].result = lab;
				params[t].start = t*step;
				params[t].end = num_vectors;
				params[t].verbose = true;
				classify_example_helper((void*) &params[t]);

				for (t=0; t<num_threads-1; t++)
					pthread_join(threads[t], NULL);
			}
#endif
		}

#ifndef WIN32
		if ( CSignal::cancel_computations() )
			SG_INFO( "prematurely stopped.           \n");
		else
#endif
			SG_DONE();
	}
	else 
		return NULL;

	return lab;
}

DREAL CSVM::classify_example(int32_t num)
{
	ASSERT(kernel);

	if (kernel->has_property(KP_LINADD) && (kernel->get_is_initialized()))
	{
		DREAL dist = kernel->compute_optimized(num);
		return (dist+get_bias());
	}
	else
	{
		DREAL dist=0;
		for(int32_t i=0; i<get_num_support_vectors(); i++)
			dist+=kernel->kernel(get_support_vector(i), num)*get_alpha(i);

		return (dist+get_bias());
	}
}


DREAL CSVM::compute_objective()
{
	int32_t n=get_num_support_vectors();

	if (labels && kernel)
	{
		objective=0;
		for (int32_t i=0; i<n; i++)
		{
			objective-=get_alpha(i)*labels->get_label(i);
			for (int32_t j=0; j<n; j++)
				objective+=0.5*get_alpha(i)*get_alpha(j)*kernel->kernel(i,j);
		}
	}
	else
		SG_ERROR( "cannot compute objective, labels or kernel not set\n");

	return objective;
}

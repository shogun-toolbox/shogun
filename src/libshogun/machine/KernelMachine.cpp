/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "machine/KernelMachine.h"
#include "lib/Signal.h"
#include "base/Parameter.h"

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct S_THREAD_PARAM
{
	CKernelMachine* kernel_machine;
	CLabels* result;
	int32_t start;
	int32_t end;
	bool verbose;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

CKernelMachine::CKernelMachine()
: CMachine(), kernel(NULL), use_batch_computation(true), use_linadd(true), use_bias(true)
{
	m_parameters->add((CSGObject**) &kernel, "kernel");
	m_parameters->add(&use_batch_computation, "use_batch_computation",
					  "Batch computation is enabled.");
	m_parameters->add(&use_linadd, "use_linadd",
					  "Linadd is enabled.");
	m_parameters->add(&use_bias, "use_bias",
					  "Bias shall be used.");
	m_parameters->add(&m_bias, "m_bias",
					  "Bias term.");
	m_parameters->add_vector(&m_alpha, &num_svs, "m_alpha",
							 "Array of coefficients alpha.");
	m_parameters->add_vector(&m_svs, &num_svs, "m_svs",
							 "Number of ``support vectors''.");

	m_bias=0.0;
	m_alpha=NULL;
	m_svs=NULL;
	num_svs=0;
}

CKernelMachine::~CKernelMachine()
{
	SG_UNREF(kernel);

	delete[] m_alpha;
	delete[] m_svs;
}

bool CKernelMachine::init_kernel_optimization()
{
	int32_t num_sv=get_num_support_vectors();

	if (kernel && kernel->has_property(KP_LINADD) && num_sv>0)
	{
		int32_t * sv_idx    = new int32_t[num_sv] ;
		float64_t* sv_weight = new float64_t[num_sv] ;

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

CLabels* CKernelMachine::classify()
{
	CLabels* lab=NULL;

	if (!kernel)
		SG_ERROR( "Kernelmachine can not proceed without kernel!\n");

	if ( kernel && kernel->get_num_vec_rhs()>0 )
	{
		int32_t num_vectors=kernel->get_num_vec_rhs();

		lab=new CLabels(num_vectors);
		SG_DEBUG( "computing output on %d test examples\n", num_vectors);

		CSignal::clear_cancel();

		if (io->get_show_progress())
			io->enable_progress();
		else
			io->disable_progress();

		if (kernel->has_property(KP_BATCHEVALUATION) &&
				get_batch_computation_enabled())
		{
			float64_t* output=new float64_t[num_vectors];
			memset(output, 0, sizeof(float64_t)*num_vectors);

			if (get_num_support_vectors()>0)
			{
				int32_t* sv_idx=new int32_t[get_num_support_vectors()];
				float64_t* sv_weight=new float64_t[get_num_support_vectors()];
				int32_t* idx=new int32_t[num_vectors];

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
				delete[] sv_idx ;
				delete[] sv_weight ;
				delete[] idx;
			}

			for (int32_t i=0; i<num_vectors; i++)
				lab->set_label(i, get_bias()+output[i]);

			delete[] output;
		}
		else
		{
			int32_t num_threads=parallel->get_num_threads();
			ASSERT(num_threads>0);

			if (num_threads < 2)
			{
				S_THREAD_PARAM params;
				params.kernel_machine=this;
				params.result=lab;
				params.start=0;
				params.end=num_vectors;
				params.verbose=true;
				classify_example_helper((void*) &params);
			}
#ifndef WIN32
			else
			{
				pthread_t* threads = new pthread_t[num_threads-1];
				S_THREAD_PARAM* params = new S_THREAD_PARAM[num_threads];
				int32_t step= num_vectors/num_threads;

				int32_t t;

				for (t=0; t<num_threads-1; t++)
				{
					params[t].kernel_machine = this;
					params[t].result = lab;
					params[t].start = t*step;
					params[t].end = (t+1)*step;
					params[t].verbose = false;
					pthread_create(&threads[t], NULL,
							CKernelMachine::classify_example_helper, (void*)&params[t]);
				}

				params[t].kernel_machine = this;
				params[t].result = lab;
				params[t].start = t*step;
				params[t].end = num_vectors;
				params[t].verbose = true;
				classify_example_helper((void*) &params[t]);

				for (t=0; t<num_threads-1; t++)
					pthread_join(threads[t], NULL);

				delete[] params;
				delete[] threads;
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

float64_t CKernelMachine::classify_example(int32_t num)
{
	ASSERT(kernel);

	if (kernel->has_property(KP_LINADD) && (kernel->get_is_initialized()))
	{
		float64_t score = kernel->compute_optimized(num);
		return score+get_bias();
	}
	else
	{
		float64_t score=0;
		for(int32_t i=0; i<get_num_support_vectors(); i++)
			score+=kernel->kernel(get_support_vector(i), num)*get_alpha(i);

		return score+get_bias();
	}
}


CLabels* CKernelMachine::classify(CFeatures* data)
{
	if (!kernel)
		SG_ERROR("No kernel assigned!\n");

	CFeatures* lhs=kernel->get_lhs();
	if (!lhs || !lhs->get_num_vectors())
	{
		SG_UNREF(lhs);
		SG_ERROR("No vectors on left hand side\n");
	}
	kernel->init(lhs, data);
	SG_UNREF(lhs);

	return classify();
}

void* CKernelMachine::classify_example_helper(void* p)
{
	S_THREAD_PARAM* params= (S_THREAD_PARAM*) p;
	CLabels* result=params->result;
	CKernelMachine* kernel_machine=params->kernel_machine;

#ifdef WIN32
	for (int32_t vec=params->start; vec<params->end; vec++)
#else
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

		result->set_label(vec, kernel_machine->classify_example(vec));
	}

	return NULL;
}

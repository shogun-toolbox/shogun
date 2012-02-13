/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/machine/KernelMachine.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parameter.h>

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

CKernelMachine::CKernelMachine() : CMachine()
{
    init();
}

CKernelMachine::CKernelMachine(CKernel* k, SGVector<float64_t> alphas,
        SGVector<int32_t> svs, float64_t b) : CMachine()
{
    init();

    int32_t num_sv=svs.vlen;
    ASSERT(num_sv == alphas.vlen);
    create_new_model(num_sv);
    set_alphas(alphas);
    set_support_vectors(svs);
    set_kernel(kernel);
    set_bias(b);
}

CKernelMachine::~CKernelMachine()
{
	SG_UNREF(kernel);
	SG_UNREF(m_custom_kernel);
	SG_UNREF(m_kernel_backup);

	SG_FREE(m_alpha.vector);
	SG_FREE(m_svs.vector);
}

void CKernelMachine::set_kernel(CKernel* k)
{
    SG_UNREF(kernel);
    SG_REF(k);
    kernel=k;
}

CKernel* CKernelMachine::get_kernel()
{
    SG_REF(kernel);
    return kernel;
}

void CKernelMachine::set_batch_computation_enabled(bool enable)
{
    use_batch_computation=enable;
}

bool CKernelMachine::get_batch_computation_enabled()
{
    return use_batch_computation;
}

void CKernelMachine::set_linadd_enabled(bool enable)
{
    use_linadd=enable;
}

bool CKernelMachine::get_linadd_enabled()
{
    return use_linadd;
}

void CKernelMachine::set_bias_enabled(bool enable_bias)
{
    use_bias=enable_bias;
}

bool CKernelMachine::get_bias_enabled()
{
    return use_bias;
}

float64_t CKernelMachine::get_bias()
{
    return m_bias;
}

void CKernelMachine::set_bias(float64_t bias)
{
    m_bias=bias;
}

int32_t CKernelMachine::get_support_vector(int32_t idx)
{
    ASSERT(m_svs.vector && idx<m_svs.vlen);
    return m_svs.vector[idx];
}

float64_t CKernelMachine::get_alpha(int32_t idx)
{
    if (!m_alpha.vector)
        SG_ERROR("No alphas set\n");
    if (idx>=m_alpha.vlen)
        SG_ERROR("Alphas index (%d) out of range (%d)\n", idx, m_svs.vlen);
    return m_alpha.vector[idx];
}

bool CKernelMachine::set_support_vector(int32_t idx, int32_t val)
{
    if (m_svs.vector && idx<m_svs.vlen)
        m_svs.vector[idx]=val;
    else
        return false;

    return true;
}

bool CKernelMachine::set_alpha(int32_t idx, float64_t val)
{
    if (m_alpha.vector && idx<m_alpha.vlen)
        m_alpha.vector[idx]=val;
    else
        return false;

    return true;
}

int32_t CKernelMachine::get_num_support_vectors()
{
    return m_svs.vlen;
}

void CKernelMachine::set_alphas(SGVector<float64_t> alphas)
{
    m_alpha = alphas;
}

void CKernelMachine::set_support_vectors(SGVector<int32_t> svs)
{
    m_svs = svs;
}

SGVector<int32_t> CKernelMachine::get_support_vectors()
{
	return m_svs;
}

SGVector<float64_t> CKernelMachine::get_alphas()
{
	return m_alpha;
}

bool CKernelMachine::create_new_model(int32_t num)
{
    m_alpha.destroy_vector();
    m_svs.destroy_vector();

    m_bias=0;

    if (num>0)
    {
        m_alpha= SGVector<float64_t>(num);
        m_svs= SGVector<int32_t>(num);
        return (m_alpha.vector!=NULL && m_svs.vector!=NULL);
    }
    else
        return true;
}

bool CKernelMachine::init_kernel_optimization()
{
	int32_t num_sv=get_num_support_vectors();

	if (kernel && kernel->has_property(KP_LINADD) && num_sv>0)
	{
		int32_t * sv_idx    = SG_MALLOC(int32_t, num_sv);
		float64_t* sv_weight = SG_MALLOC(float64_t, num_sv);

		for(int32_t i=0; i<num_sv; i++)
		{
			sv_idx[i]    = get_support_vector(i) ;
			sv_weight[i] = get_alpha(i) ;
		}

		bool ret = kernel->init_optimization(num_sv, sv_idx, sv_weight) ;

		SG_FREE(sv_idx);
		SG_FREE(sv_weight);

		if (!ret)
			SG_ERROR( "initialization of kernel optimization failed\n");

		return ret;
	}
	else
		SG_ERROR( "initialization of kernel optimization failed\n");

	return false;
}

CLabels* CKernelMachine::apply()
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
			float64_t* output=SG_MALLOC(float64_t, num_vectors);
			memset(output, 0, sizeof(float64_t)*num_vectors);

			if (get_num_support_vectors()>0)
			{
				int32_t* sv_idx=SG_MALLOC(int32_t, get_num_support_vectors());
				float64_t* sv_weight=SG_MALLOC(float64_t, get_num_support_vectors());
				int32_t* idx=SG_MALLOC(int32_t, num_vectors);

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
				SG_FREE(sv_idx);
				SG_FREE(sv_weight);
				SG_FREE(idx);
			}

			for (int32_t i=0; i<num_vectors; i++)
				lab->set_label(i, get_bias()+output[i]);

			SG_FREE(output);
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
				apply_helper((void*) &params);
			}
#ifdef HAVE_PTHREAD
			else
			{
				pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
				S_THREAD_PARAM* params = SG_MALLOC(S_THREAD_PARAM, num_threads);
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
							CKernelMachine::apply_helper, (void*)&params[t]);
				}

				params[t].kernel_machine = this;
				params[t].result = lab;
				params[t].start = t*step;
				params[t].end = num_vectors;
				params[t].verbose = true;
				apply_helper((void*) &params[t]);

				for (t=0; t<num_threads-1; t++)
					pthread_join(threads[t], NULL);

				SG_FREE(params);
				SG_FREE(threads);
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

float64_t CKernelMachine::apply(int32_t num)
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


CLabels* CKernelMachine::apply(CFeatures* data)
{
	if (m_kernel_backup)
	{
		SG_ERROR("CKernelMachine::apply(CFeatures*) cannot be called when "
				"data_lock was called before - only apply() and apply(int32_t)."
				" Call data_unlock to allow.");
	}

	if (!kernel)
		SG_ERROR("No kernel assigned!\n");

	CFeatures* lhs=kernel->get_lhs();
	if (!lhs)
		SG_ERROR("%s: No left hand side specified\n", get_name());

	if (!lhs->get_num_vectors())
	{
		SG_ERROR("%s: No vectors on left hand side (%s). This is probably due to"
				" an implementation error in %s, where it was forgotten to set "
				"the data (m_svs) indices\n", get_name(),
				data->get_name());
	}

	kernel->init(lhs, data);
	SG_UNREF(lhs);

	return apply();
}

void* CKernelMachine::apply_helper(void* p)
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

		result->set_label(vec, kernel_machine->apply(vec));
	}

	return NULL;
}

void CKernelMachine::store_model_features()
{
	if (!kernel)
		SG_ERROR("kernel is needed to store SV features.\n");

	CFeatures* lhs=kernel->get_lhs();
	CFeatures* rhs=kernel->get_rhs();

	if (!lhs)
		SG_ERROR("kernel lhs is needed to store SV features.\n");

	/* copy sv feature data */
	CFeatures* sv_features=lhs->copy_subset(m_svs);
	SG_UNREF(lhs);

	/* now sv indices are just the identity */
	m_svs.range_fill();

	/* set new lhs to kernel */
	kernel->init(sv_features, rhs);

	SG_UNREF(rhs);
}

bool CKernelMachine::train_locked(SGVector<index_t> indices)
{
	if (!is_data_locked())
		SG_ERROR("CKernelMachine::train_locked() call data_lock() before!\n");

	/* this is asusmed here */
	ASSERT(m_custom_kernel==kernel);

	/* set custom kernel subset of data to train on (copies because CSubset
	 * will delete vetors at the end)*/
	SGVector<index_t> row_inds=SGVector<index_t>(indices);
	row_inds.vector=CMath::clone_vector(indices.vector, indices.vlen);
	m_custom_kernel->set_row_subset(new CSubset(row_inds));
	SGVector<index_t> col_inds=SGVector<index_t>(indices);
	col_inds.vector=CMath::clone_vector(indices.vector, indices.vlen);
	m_custom_kernel->set_col_subset(new CSubset(col_inds));
//	SG_PRINT("training matrix:\n");
//	m_custom_kernel->print_kernel_matrix("\t");

	/* set corresponding labels subset */
	SGVector<index_t> label_inds=SGVector<index_t>(indices);
	label_inds.vector=CMath::clone_vector(indices.vector, indices.vlen);
	labels->set_subset(new CSubset(label_inds));
//	SGVector<float64_t> temp=labels->get_labels_copy();
//	CMath::display_vector(temp.vector, temp.vlen, "training labels");
//	temp.destroy_vector();

	/* dont do train because model should not be stored (no acutal features)
	 * and train does data_unlock */
//	SG_PRINT("calling train_machine\n");
	bool result=train_machine();
//	SG_PRINT("done train_machine\n");

//	CMath::display_vector(get_support_vectors().vector, get_num_support_vectors(), "sv indices");

	/* set col subset of kernel to contain all elements */
	m_custom_kernel->remove_col_subset();
//	SG_PRINT("matrix after training:\n");
//	m_custom_kernel->print_kernel_matrix("\t");

	/* remove label subset after training */
	labels->remove_subset();

	return result;
}

CLabels* CKernelMachine::apply_locked(SGVector<index_t> indices)
{
	if (!is_data_locked())
		SG_ERROR("CKernelMachine::apply_locked() call data_lock() before!\n");

	/* TODO parallelize? */
	SGVector<float64_t> output(indices.vlen);
	for (index_t i=0; i<indices.vlen; ++i)
		output.vector[i]=apply(indices.vector[i]);

	return new CLabels(output);
}

void CKernelMachine::data_lock(CFeatures* features, CLabels* labs)
{
	/* init kernel with data */
	kernel->init(features, features);

	/* backup reference to old kernel */
	SG_UNREF(m_kernel_backup)
	m_kernel_backup=kernel;
	SG_REF(m_kernel_backup);

	/* unref possible old custom kernel */
	SG_UNREF(m_custom_kernel);

	/* create custom kernel matrix from current kernel */
//	SG_PRINT("computing kernel matrix for %s\n", kernel->get_name());
	m_custom_kernel=new CCustomKernel(kernel);
	SG_REF(m_custom_kernel);
//	SG_PRINT("done\n");

	/* replace kernel by custom kernel */
	SG_UNREF(kernel);
	kernel=m_custom_kernel;
	SG_REF(kernel);

	/* dont forget to call superclass method */
	CMachine::data_lock(features, labs);
}

void CKernelMachine::data_unlock()
{
	SG_UNREF(m_custom_kernel);
	m_custom_kernel=NULL;

	/* restore original kernel, possibly delete created one */
	if (m_kernel_backup)
	{
		/* check if kernel was created in train_locked */
		if (kernel!=m_kernel_backup)
			SG_UNREF(kernel);

		kernel=m_kernel_backup;
		m_kernel_backup=NULL;
	}

	/* dont forget to call superclass method */
	CMachine::data_unlock();
}

void CKernelMachine::init()
{
	m_bias=0.0;
	kernel=NULL;
	m_custom_kernel=NULL;
	m_kernel_backup=NULL;
	use_batch_computation=true;
	use_linadd=true;
	use_bias=true;

	SG_ADD((CSGObject**) &kernel, "kernel", "", MS_AVAILABLE);
	SG_ADD((CSGObject**) &m_custom_kernel, "custom_kernel", "Custom kernel for"
			" data lock", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_kernel_backup, "kernel_backup",
			"Kernel backup for data lock", MS_NOT_AVAILABLE);
	SG_ADD(&use_batch_computation, "use_batch_computation",
			"Batch computation is enabled.", MS_NOT_AVAILABLE);
	SG_ADD(&use_linadd, "use_linadd", "Linadd is enabled.", MS_NOT_AVAILABLE);
	SG_ADD(&use_bias, "use_bias", "Bias shall be used.", MS_NOT_AVAILABLE);
	SG_ADD(&m_bias, "m_bias", "Bias term.", MS_NOT_AVAILABLE);
	SG_ADD(&m_alpha, "m_alpha", "Array of coefficients alpha.",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_svs, "m_svs", "Number of ``support vectors''.", MS_NOT_AVAILABLE);
}

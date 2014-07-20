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
#include <shogun/labels/RegressionLabels.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/ParameterMap.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/labels/Labels.h>

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct S_THREAD_PARAM_KERNEL_MACHINE
{
	CKernelMachine* kernel_machine;
	float64_t* result;
	int32_t start;
	int32_t end;

	/* if non-null, start and end correspond to indices in this vector */
	index_t* indices;
	index_t indices_len;
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
    ASSERT(num_sv == alphas.vlen)
    create_new_model(num_sv);
    set_alphas(alphas);
    set_support_vectors(svs);
    set_kernel(kernel);
    set_bias(b);
}

CKernelMachine::CKernelMachine(CKernelMachine* machine) : CMachine()
{
	init();

	SGVector<float64_t> alphas = machine->get_alphas().clone();
	SGVector<int32_t> svs = machine->get_support_vectors().clone();
	float64_t bias = machine->get_bias();
	CKernel* ker = machine->get_kernel();

	int32_t num_sv = svs.vlen;
	create_new_model(num_sv);
	set_alphas(alphas);
	set_support_vectors(svs);
	set_bias(bias);
	set_kernel(ker);
}

CKernelMachine::~CKernelMachine()
{
	SG_UNREF(kernel);
	SG_UNREF(m_custom_kernel);
	SG_UNREF(m_kernel_backup);
}

void CKernelMachine::set_kernel(CKernel* k)
{
	SG_REF(k);
	SG_UNREF(kernel);
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
    ASSERT(m_svs.vector && idx<m_svs.vlen)
    return m_svs.vector[idx];
}

float64_t CKernelMachine::get_alpha(int32_t idx)
{
    if (!m_alpha.vector)
        SG_ERROR("No alphas set\n")
    if (idx>=m_alpha.vlen)
        SG_ERROR("Alphas index (%d) out of range (%d)\n", idx, m_svs.vlen)
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
    m_alpha=SGVector<float64_t>();
    m_svs=SGVector<int32_t>();

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
			SG_ERROR("initialization of kernel optimization failed\n")

		return ret;
	}
	else
		SG_ERROR("initialization of kernel optimization failed\n")

	return false;
}

CRegressionLabels* CKernelMachine::apply_regression(CFeatures* data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return new CRegressionLabels(outputs);
}

CBinaryLabels* CKernelMachine::apply_binary(CFeatures* data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return new CBinaryLabels(outputs);
}

SGVector<float64_t> CKernelMachine::apply_get_outputs(CFeatures* data)
{
	SG_DEBUG("entering %s::apply_get_outputs(%s at %p)\n",
			get_name(), data ? data->get_name() : "NULL", data);

	REQUIRE(kernel, "%s::apply_get_outputs(): No kernel assigned!\n")

	if (!kernel->get_num_vec_lhs())
	{
		SG_ERROR("%s: No vectors on left hand side (%s). This is probably due to"
				" an implementation error in %s, where it was forgotten to set "
				"the data (m_svs) indices\n", get_name(),
				data->get_name());
	}

	if (data)
	{
		CFeatures* lhs=kernel->get_lhs();
		REQUIRE(lhs, "%s::apply_get_outputs(): No left hand side specified\n",
				get_name());
		kernel->init(lhs, data);
		SG_UNREF(lhs);
	}

	/* using the features to get num vectors is far safer than using the kernel
	 * since SHOGUNs kernel num_rhs/num_lhs is buggy (CombinedKernel for ex.)
	 * Might be worth investigating why
	 * kernel->get_num_rhs() != rhs->get_num_vectors()
	 * However, the below version works
	 * TODO Heiko Strathmann
	 */
	CFeatures* rhs=kernel->get_rhs();
	int32_t num_vectors=rhs ? rhs->get_num_vectors() : kernel->get_num_vec_rhs();
	SG_UNREF(rhs)

	SGVector<float64_t> output(num_vectors);

	if (kernel->get_num_vec_rhs()>0)
	{
		SG_DEBUG("computing output on %d test examples\n", num_vectors)

		CSignal::clear_cancel();

		if (io->get_show_progress())
			io->enable_progress();
		else
			io->disable_progress();

		if (kernel->has_property(KP_BATCHEVALUATION) &&
				get_batch_computation_enabled())
		{
			output.zero();
			SG_DEBUG("Batch evaluation enabled\n")
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
						output.vector, get_num_support_vectors(), sv_idx, sv_weight);
				SG_FREE(sv_idx);
				SG_FREE(sv_weight);
				SG_FREE(idx);
			}

			for (int32_t i=0; i<num_vectors; i++)
				output[i] = get_bias() + output[i];

		}
		else
		{
			int32_t num_threads=parallel->get_num_threads();
			ASSERT(num_threads>0)

			if (num_threads < 2)
			{
				S_THREAD_PARAM_KERNEL_MACHINE params;
				params.kernel_machine=this;
				params.result = output.vector;
				params.start=0;
				params.end=num_vectors;
				params.verbose=true;
				params.indices = NULL;
				params.indices_len = 0;
				apply_helper((void*) &params);
			}
#ifdef HAVE_PTHREAD
			else
			{
				pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
				S_THREAD_PARAM_KERNEL_MACHINE* params = SG_MALLOC(S_THREAD_PARAM_KERNEL_MACHINE, num_threads);
				int32_t step= num_vectors/num_threads;

				int32_t t;

				for (t=0; t<num_threads-1; t++)
				{
					params[t].kernel_machine = this;
					params[t].result = output.vector;
					params[t].start = t*step;
					params[t].end = (t+1)*step;
					params[t].verbose = false;
					params[t].indices = NULL;
					params[t].indices_len = 0;
					pthread_create(&threads[t], NULL,
							CKernelMachine::apply_helper, (void*)&params[t]);
				}

				params[t].kernel_machine = this;
				params[t].result = output.vector;
				params[t].start = t*step;
				params[t].end = num_vectors;
				params[t].verbose = true;
				params[t].indices = NULL;
				params[t].indices_len = 0;
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
			SG_INFO("prematurely stopped.           \n")
		else
#endif
			SG_DONE()
	}

	SG_DEBUG("leaving %s::apply_get_outputs(%s at %p)\n",
			get_name(), data ? data->get_name() : "NULL", data);

	return output;
}

float64_t CKernelMachine::apply_one(int32_t num)
{
	ASSERT(kernel)

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

void* CKernelMachine::apply_helper(void* p)
{
	S_THREAD_PARAM_KERNEL_MACHINE* params = (S_THREAD_PARAM_KERNEL_MACHINE*) p;
	float64_t* result = params->result;
	CKernelMachine* kernel_machine = params->kernel_machine;

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
				SG_SPROGRESS(v, 0.0, num_vectors-1)
		}

		/* eventually use index mapping if exists */
		index_t idx=params->indices ? params->indices[vec] : vec;
		result[vec] = kernel_machine->apply_one(idx);
	}

	return NULL;
}

void CKernelMachine::store_model_features()
{
	if (!kernel)
		SG_ERROR("kernel is needed to store SV features.\n")

	CFeatures* lhs=kernel->get_lhs();
	CFeatures* rhs=kernel->get_rhs();

	if (!lhs)
		SG_ERROR("kernel lhs is needed to store SV features.\n")

	/* copy sv feature data */
	CFeatures* sv_features=lhs->copy_subset(m_svs);
	SG_UNREF(lhs);

	/* set new lhs to kernel */
	kernel->init(sv_features, rhs);

	/* unref rhs */
	SG_UNREF(rhs);

	/* was SG_REF'ed by copy_subset */
	SG_UNREF(sv_features);

	/* now sv indices are just the identity */
	m_svs.range_fill();

}

bool CKernelMachine::train_locked(SGVector<index_t> indices)
{
	SG_DEBUG("entering %s::train_locked()\n", get_name())
	if (!is_data_locked())
		SG_ERROR("CKernelMachine::train_locked() call data_lock() before!\n")

	/* this is asusmed here */
	ASSERT(m_custom_kernel==kernel)

	/* since its not easily possible to controll the row subsets of the custom
	 * kernel from outside, we enforce that there is only one row subset by
	 * removing all of them. Otherwise, they would add up in the stack until
	 * an error occurs */
	m_custom_kernel->remove_all_row_subsets();

	/* set custom kernel subset of data to train on */
	m_custom_kernel->add_row_subset(indices);
	m_custom_kernel->add_col_subset(indices);

	/* set corresponding labels subset */
	m_labels->add_subset(indices);

	/* dont do train because model should not be stored (no acutal features)
	 * and train does data_unlock */
	bool result=train_machine();

	/* remove last col subset of custom kernel */
	m_custom_kernel->remove_col_subset();

	/* remove label subset after training */
	m_labels->remove_subset();

	SG_DEBUG("leaving %s::train_locked()\n", get_name())
	return result;
}

CBinaryLabels* CKernelMachine::apply_locked_binary(SGVector<index_t> indices)
{
	SGVector<float64_t> outputs = apply_locked_get_output(indices);
	return new CBinaryLabels(outputs);
}

CRegressionLabels* CKernelMachine::apply_locked_regression(
		SGVector<index_t> indices)
{
	SGVector<float64_t> outputs = apply_locked_get_output(indices);
	return new CRegressionLabels(outputs);
}

SGVector<float64_t> CKernelMachine::apply_locked_get_output(
		SGVector<index_t> indices)
{
	if (!is_data_locked())
		SG_ERROR("CKernelMachine::apply_locked() call data_lock() before!\n")

	/* we are working on a custom kernel here */
	ASSERT(m_custom_kernel==kernel)

	int32_t num_inds=indices.vlen;
	SGVector<float64_t> output(num_inds);

	CSignal::clear_cancel();

	if (io->get_show_progress())
		io->enable_progress();
	else
		io->disable_progress();

	/* custom kernel never has batch evaluation property so dont do this here */
	int32_t num_threads=parallel->get_num_threads();
	ASSERT(num_threads>0)

	if (num_threads<2)
	{
		S_THREAD_PARAM_KERNEL_MACHINE params;
		params.kernel_machine=this;
		params.result=output.vector;

		/* use the parameter index vector */
		params.start=0;
		params.end=num_inds;
		params.indices=indices.vector;
		params.indices_len=indices.vlen;

		params.verbose=true;
		apply_helper((void*) &params);
	}
#ifdef HAVE_PTHREAD
	else
	{
		pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
		S_THREAD_PARAM_KERNEL_MACHINE* params=SG_MALLOC(S_THREAD_PARAM_KERNEL_MACHINE, num_threads);
		int32_t step= num_inds/num_threads;

		int32_t t;
		for (t=0; t<num_threads-1; t++)
		{
			params[t].kernel_machine=this;
			params[t].result=output.vector;

			/* use the parameter index vector */
			params[t].start=t*step;
			params[t].end=(t+1)*step;
			params[t].indices=indices.vector;
			params[t].indices_len=indices.vlen;

			params[t].verbose=false;
			pthread_create(&threads[t], NULL, CKernelMachine::apply_helper,
					(void*)&params[t]);
		}

		params[t].kernel_machine=this;
		params[t].result=output.vector;

		/* use the parameter index vector */
		params[t].start=t*step;
		params[t].end=num_inds;
		params[t].indices=indices.vector;
		params[t].indices_len=indices.vlen;

		params[t].verbose=true;
		apply_helper((void*) &params[t]);

		for (t=0; t<num_threads-1; t++)
			pthread_join(threads[t], NULL);

		SG_FREE(params);
		SG_FREE(threads);
	}
#endif

#ifndef WIN32
	if ( CSignal::cancel_computations() )
		SG_INFO("prematurely stopped.\n")
	else
#endif
		SG_DONE()

	return output;
}

void CKernelMachine::data_lock(CLabels* labs, CFeatures* features)
{
	if ( !kernel )
		SG_ERROR("The kernel is not initialized\n")
	if (kernel->has_property(KP_KERNCOMBINATION))
		SG_ERROR("Locking is not supported (yet) with combined kernel. Please disable it in cross validation")

	/* init kernel with data */
	kernel->init(features, features);

	/* backup reference to old kernel */
	SG_UNREF(m_kernel_backup)
	m_kernel_backup=kernel;
	SG_REF(m_kernel_backup);

	/* unref possible old custom kernel */
	SG_UNREF(m_custom_kernel);

	/* create custom kernel matrix from current kernel */
	m_custom_kernel=new CCustomKernel(kernel);
	SG_REF(m_custom_kernel);

	/* replace kernel by custom kernel */
	SG_UNREF(kernel);
	kernel=m_custom_kernel;
	SG_REF(kernel);

	/* dont forget to call superclass method */
	CMachine::data_lock(labs, features);
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

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
		new SGParamInfo("custom_kernel", CT_SCALAR, ST_NONE, PT_SGOBJECT, 1),
		new SGParamInfo()
	);

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
		new SGParamInfo("kernel_backup", CT_SCALAR, ST_NONE, PT_SGOBJECT, 1),
		new SGParamInfo()
	);
	m_parameter_map->finalize_map();
}

bool CKernelMachine::supports_locking() const
{
	return true;
}


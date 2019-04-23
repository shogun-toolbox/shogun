/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Giovanni De Toni, Viktor Gal, Evgeniy Andreev, Weijie Lin,
 *          Fernando Iglesias, Thoralf Klein
 */

#include <rxcpp/rx-lite.hpp>
#include <shogun/base/progress.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/machine/KernelMachine.h>

#ifdef HAVE_OPENMP
#include <omp.h>

#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct S_THREAD_PARAM_KERNEL_MACHINE
{
	KernelMachine* kernel_machine;
	float64_t* result;
	int32_t start;
	int32_t end;

	/* if non-null, start and end correspond to indices in this vector */
	index_t* indices;
	index_t indices_len;
	bool verbose;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

KernelMachine::KernelMachine() : Machine()
{
    init();
}

KernelMachine::KernelMachine(std::shared_ptr<Kernel> k, SGVector<float64_t> alphas,
        SGVector<int32_t> svs, float64_t b) : Machine()
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

KernelMachine::KernelMachine(std::shared_ptr<KernelMachine> machine) : Machine()
{
	init();

	SGVector<float64_t> alphas = machine->get_alphas().clone();
	SGVector<int32_t> svs = machine->get_support_vectors().clone();
	float64_t bias = machine->get_bias();
	auto ker = machine->get_kernel();

	int32_t num_sv = svs.vlen;
	create_new_model(num_sv);
	set_alphas(alphas);
	set_support_vectors(svs);
	set_bias(bias);
	set_kernel(ker);
}

KernelMachine::~KernelMachine()
{
}

void KernelMachine::set_kernel(std::shared_ptr<Kernel> k)
{
	kernel=k;
}

std::shared_ptr<Kernel> KernelMachine::get_kernel()
{

    return kernel;
}

void KernelMachine::set_batch_computation_enabled(bool enable)
{
    use_batch_computation=enable;
}

bool KernelMachine::get_batch_computation_enabled()
{
    return use_batch_computation;
}

void KernelMachine::set_linadd_enabled(bool enable)
{
    use_linadd=enable;
}

bool KernelMachine::get_linadd_enabled()
{
    return use_linadd;
}

void KernelMachine::set_bias_enabled(bool enable_bias)
{
    use_bias=enable_bias;
}

bool KernelMachine::get_bias_enabled()
{
    return use_bias;
}

float64_t KernelMachine::get_bias()
{
    return m_bias;
}

void KernelMachine::set_bias(float64_t bias)
{
    m_bias=bias;
}

int32_t KernelMachine::get_support_vector(int32_t idx)
{
    ASSERT(m_svs.vector && idx<m_svs.vlen)
    return m_svs.vector[idx];
}

float64_t KernelMachine::get_alpha(int32_t idx)
{
    if (!m_alpha.vector)
        SG_ERROR("No alphas set\n")
    if (idx>=m_alpha.vlen)
        SG_ERROR("Alphas index (%d) out of range (%d)\n", idx, m_svs.vlen)
    return m_alpha.vector[idx];
}

bool KernelMachine::set_support_vector(int32_t idx, int32_t val)
{
    if (m_svs.vector && idx<m_svs.vlen)
        m_svs.vector[idx]=val;
    else
        return false;

    return true;
}

bool KernelMachine::set_alpha(int32_t idx, float64_t val)
{
    if (m_alpha.vector && idx<m_alpha.vlen)
        m_alpha.vector[idx]=val;
    else
        return false;

    return true;
}

int32_t KernelMachine::get_num_support_vectors()
{
    return m_svs.vlen;
}

void KernelMachine::set_alphas(SGVector<float64_t> alphas)
{
    m_alpha = alphas;
}

void KernelMachine::set_support_vectors(SGVector<int32_t> svs)
{
    m_svs = svs;
}

SGVector<int32_t> KernelMachine::get_support_vectors()
{
	return m_svs;
}

SGVector<float64_t> KernelMachine::get_alphas()
{
	return m_alpha;
}

bool KernelMachine::create_new_model(int32_t num)
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

bool KernelMachine::init_kernel_optimization()
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

std::shared_ptr<RegressionLabels> KernelMachine::apply_regression(std::shared_ptr<Features> data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return std::make_shared<RegressionLabels>(outputs);
}

std::shared_ptr<BinaryLabels> KernelMachine::apply_binary(std::shared_ptr<Features> data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return std::make_shared<BinaryLabels>(outputs);
}

SGVector<float64_t> KernelMachine::apply_get_outputs(std::shared_ptr<Features> data)
{
	SG_DEBUG("entering %s::apply_get_outputs(%s at %p)\n",
			get_name(), data ? data->get_name() : "NULL", data.get());

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
		auto lhs=kernel->get_lhs();
		REQUIRE(lhs, "%s::apply_get_outputs(): No left hand side specified\n",
				get_name());
		kernel->init(lhs, data);

	}

	/* using the features to get num vectors is far safer than using the kernel
	 * since SHOGUNs kernel num_rhs/num_lhs is buggy (CombinedKernel for ex.)
	 * Might be worth investigating why
	 * kernel->get_num_rhs() != rhs->get_num_vectors()
	 * However, the below version works
	 * TODO Heiko Strathmann
	 */
	auto rhs=kernel->get_rhs();
	int32_t num_vectors=rhs ? rhs->get_num_vectors() : kernel->get_num_vec_rhs();


	SGVector<float64_t> output(num_vectors);

	if (kernel->get_num_vec_rhs()>0)
	{
		SG_DEBUG("computing output on %d test examples\n", num_vectors)

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
			auto pb = SG_PROGRESS(range(num_vectors));
			int32_t num_threads;
			int64_t step;
#pragma omp parallel shared(num_threads, step)
			{

#ifdef HAVE_OPENMP
#pragma omp single
				{
					num_threads = omp_get_num_threads();
					step = num_vectors / num_threads;
					num_threads--;
				}
				int32_t thread_num = omp_get_thread_num();
#else
				num_threads = 0;
				step = num_vectors;
				int32_t thread_num = 0;
#endif
				int32_t start = thread_num * step;
				int32_t end = (thread_num == num_threads)
				                  ? num_vectors
				                  : (thread_num + 1) * step;

				for (int32_t vec = start; vec < end; vec++)
				{
					COMPUTATION_CONTROLLERS
					pb.print_progress();

					ASSERT(kernel)
					if (kernel->has_property(KP_LINADD) &&
					    (kernel->get_is_initialized()))
					{
						float64_t score = kernel->compute_optimized(vec);
						output[vec] = score + get_bias();
					}
					else
					{
						float64_t score = 0;
						for (int32_t i = 0; i < get_num_support_vectors(); i++)
							score +=
							    kernel->kernel(get_support_vector(i), vec) *
							    get_alpha(i);
						output[vec] = score + get_bias();
					}
				}
			}
			pb.complete();
		}
	}

	SG_DEBUG("leaving %s::apply_get_outputs(%s at %p)\n",
			get_name(), data ? data->get_name() : "NULL", data.get());

	return output;
}

void KernelMachine::store_model_features()
{
	if (!kernel)
		SG_ERROR("kernel is needed to store SV features.\n")

	auto lhs=kernel->get_lhs();
	auto rhs=kernel->get_rhs();

	if (!lhs)
		SG_ERROR("kernel lhs is needed to store SV features.\n")

	/* copy sv feature data */
	auto sv_features=lhs->copy_subset(m_svs);


	/* set new lhs to kernel */
	kernel->init(sv_features, rhs);

	/* unref rhs */


	/* was SG_REF'ed by copy_subset */


	/* now sv indices are just the identity */
	m_svs.range_fill();

}

bool KernelMachine::train_locked(SGVector<index_t> indices)
{
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
	bool result = Machine::train_locked();
	/* remove last col subset of custom kernel */
	m_custom_kernel->remove_col_subset();

	/* remove label subset after training */
	m_labels->remove_subset();

	return result;
}

std::shared_ptr<BinaryLabels> KernelMachine::apply_locked_binary(SGVector<index_t> indices)
{
	SGVector<float64_t> outputs = apply_locked_get_output(indices);
	return std::make_shared<BinaryLabels>(outputs);
}

std::shared_ptr<RegressionLabels> KernelMachine::apply_locked_regression(
		SGVector<index_t> indices)
{
	SGVector<float64_t> outputs = apply_locked_get_output(indices);
	return std::make_shared<RegressionLabels>(outputs);
}

SGVector<float64_t> KernelMachine::apply_locked_get_output(
		SGVector<index_t> indices)
{
	if (!is_data_locked())
		SG_ERROR("KernelMachine::apply_locked() call data_lock() before!\n")

	/* we are working on a custom kernel here */
	ASSERT(m_custom_kernel==kernel)

	int32_t num_inds=indices.vlen;
	SGVector<float64_t> output(num_inds);

	if (io->get_show_progress())
		io->enable_progress();
	else
		io->disable_progress();

	/* custom kernel never has batch evaluation property so dont do this here */
	auto pb = SG_PROGRESS(range(0, num_inds));
	int32_t num_threads;
	int64_t step;
#pragma omp parallel shared(num_threads, step)
	{
#ifdef HAVE_OPENMP
#pragma omp single
		{
			num_threads = omp_get_num_threads();
			step = num_inds / num_threads;
			num_threads--;
		}
		int32_t thread_num = omp_get_thread_num();
#else
		num_threads = 0;
		step = num_inds;
		int32_t thread_num = 0;
#endif
		int32_t start = thread_num * step;
		int32_t end =
		    (thread_num == num_threads) ? num_inds : (thread_num + 1) * step;

		for (int32_t vec = start; vec < end; vec++)
		{
			COMPUTATION_CONTROLLERS
			pb.print_progress();
			index_t index = indices[vec];
			ASSERT(kernel)
			if (kernel->has_property(KP_LINADD) &&
			    (kernel->get_is_initialized()))
			{
				float64_t score = kernel->compute_optimized(index);
				output[vec] = score + get_bias();
			}
			else
			{
				float64_t score = 0;
				for (int32_t i = 0; i < get_num_support_vectors(); i++)
					score += kernel->kernel(get_support_vector(i), index) *
					         get_alpha(i);

				output[vec] = score + get_bias();
			}
		}
	}
	pb.complete();

	return output;
}

float64_t KernelMachine::apply_one(int32_t num)
{
	ASSERT(kernel)

	if (kernel->has_property(KP_LINADD) && (kernel->get_is_initialized()))
	{
		float64_t score = kernel->compute_optimized(num);
		return score + get_bias();
	}
	else
	{
		float64_t score = 0;
		for (int32_t i = 0; i < get_num_support_vectors(); i++)
			score += kernel->kernel(get_support_vector(i), num) * get_alpha(i);

		return score + get_bias();
	}
}

void KernelMachine::data_lock(std::shared_ptr<Labels> labs, std::shared_ptr<Features> features)
{
	if ( !kernel )
		SG_ERROR("The kernel is not initialized\n")
	if (kernel->has_property(KP_KERNCOMBINATION))
		SG_ERROR("Locking is not supported (yet) with combined kernel. Please disable it in cross validation")

	/* init kernel with data */
	kernel->init(features, features);

	/* backup reference to old kernel */

	m_kernel_backup=kernel;


	/* unref possible old custom kernel */


	/* create custom kernel matrix from current kernel */
	m_custom_kernel=std::make_shared<CustomKernel>(kernel);


	/* replace kernel by custom kernel */

	kernel=m_custom_kernel;


	/* dont forget to call superclass method */
	Machine::data_lock(labs, features);
}

void KernelMachine::data_unlock()
{

	m_custom_kernel=NULL;

	/* restore original kernel, possibly delete created one */
	if (m_kernel_backup)
	{
		/* check if kernel was created in train_locked */
		if (kernel!=m_kernel_backup)


		kernel=m_kernel_backup;
		m_kernel_backup=NULL;
	}

	/* dont forget to call superclass method */
	Machine::data_unlock();
}

void KernelMachine::init()
{
	m_bias=0.0;
	kernel=NULL;
	m_custom_kernel=NULL;
	m_kernel_backup=NULL;
	use_batch_computation=true;
	use_linadd=true;
	use_bias=true;

	SG_ADD(&kernel, "kernel", "", ParameterProperties::HYPER);
	SG_ADD((std::shared_ptr<Kernel>*) &m_custom_kernel, "custom_kernel", "Custom kernel for"
			" data lock");
	SG_ADD(&m_kernel_backup, "kernel_backup",
			"Kernel backup for data lock");
	SG_ADD(&use_batch_computation, "use_batch_computation",
			"Batch computation is enabled.");
	SG_ADD(&use_linadd, "use_linadd", "Linadd is enabled.");
	SG_ADD(&use_bias, "use_bias", "Bias shall be used.");
	SG_ADD(&m_bias, "m_bias", "Bias term.");
	SG_ADD(&m_alpha, "m_alpha", "Array of coefficients alpha.");
	SG_ADD(&m_svs, "m_svs", "Number of ``support vectors''.");
}

bool KernelMachine::supports_locking() const
{
	return true;
}

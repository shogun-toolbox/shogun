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
#include <utility>

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

KernelMachine::KernelMachine(const std::shared_ptr<Kernel>& k, SGVector<float64_t> alphas,
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

KernelMachine::KernelMachine(const std::shared_ptr<KernelMachine>& machine) : Machine()
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
	kernel=std::move(k);
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
        error("No alphas set");
    if (idx>=m_alpha.vlen)
        error("Alphas index ({}) out of range ({})", idx, m_svs.vlen);
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
			error("initialization of kernel optimization failed");

		return ret;
	}
	else
		error("initialization of kernel optimization failed");

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

SGVector<float64_t> KernelMachine::apply_get_outputs(const std::shared_ptr<Features>& data)
{
	SG_TRACE("entering {}::apply_get_outputs({} at {})",
			get_name(), data ? data->get_name() : "NULL", fmt::ptr(data.get()));

	require(kernel, "{}::apply_get_outputs(): No kernel assigned!");

	if (!kernel->get_num_vec_lhs())
	{
		error("{}: No vectors on left hand side ({}). This is probably due to"
				" an implementation error in {}, where it was forgotten to set "
				"the data (m_svs) indices", get_name(),
				data->get_name());
	}

	if (data)
	{
		auto lhs=kernel->get_lhs();
		require(lhs, "{}::apply_get_outputs(): No left hand side specified",
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
		SG_DEBUG("computing output on {} test examples", num_vectors)

		if (env()->io()->get_show_progress())
			env()->io()->enable_progress();
		else
			env()->io()->disable_progress();

		if (kernel->has_property(KP_BATCHEVALUATION) &&
				get_batch_computation_enabled())
		{
			output.zero();
			SG_DEBUG("Batch evaluation enabled")
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

	SG_TRACE("leaving {}::apply_get_outputs({} at {})",
			get_name(), data ? data->get_name() : "NULL", fmt::ptr(data.get()));

	return output;
}

void KernelMachine::store_model_features()
{
	if (!kernel)
		error("kernel is needed to store SV features.");

	auto lhs=kernel->get_lhs();
	auto rhs=kernel->get_rhs();

	if (!lhs)
		error("kernel lhs is needed to store SV features.");

	/* copy sv feature data */
	auto sv_features=lhs->copy_subset(m_svs);


	/* set new lhs to kernel */
	kernel->init(sv_features, rhs);

	/* unref rhs */


	/* was SG_REF'ed by copy_subset */


	/* now sv indices are just the identity */
	m_svs.range_fill();

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

void KernelMachine::init()
{
	m_bias=0.0;
	kernel=NULL;
	use_batch_computation=true;
	use_linadd=true;
	use_bias=true;

	SG_ADD(&kernel, "kernel", "", ParameterProperties::HYPER);
	SG_ADD(&use_batch_computation, "use_batch_computation",
			"Batch computation is enabled.", ParameterProperties::SETTING);
	SG_ADD(&use_linadd, "use_linadd", "Linadd is enabled.", ParameterProperties::SETTING);
	SG_ADD(&use_bias, "use_bias", "Bias shall be used.", ParameterProperties::SETTING);
	SG_ADD(&m_bias, "m_bias", "Bias term.", ParameterProperties::MODEL);
	SG_ADD(&m_alpha, "m_alpha", "Array of coefficients alpha.", ParameterProperties::MODEL);
	SG_ADD(&m_svs, "m_svs", "Number of ``support vectors''.", ParameterProperties::MODEL);
}


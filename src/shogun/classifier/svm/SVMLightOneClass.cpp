/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Viktor Gal,
 *          Bjoern Esser, Evangelos Anagnostopoulos
 */

#include <shogun/lib/config.h>

#ifdef USE_SVMLIGHT

#include <shogun/io/SGIO.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/Signal.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/classifier/svm/SVMLightOneClass.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/kernel/CombinedKernel.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}
#endif

#include <shogun/base/Parallel.h>

#include <utility>

using namespace shogun;

SVMLightOneClass::SVMLightOneClass(float64_t C, std::shared_ptr<Kernel> k)
: SVMLight()
{
	set_C(C,C);
	set_kernel(std::move(k));
}

SVMLightOneClass::SVMLightOneClass()
: SVMLight()
{
}

bool SVMLightOneClass::train_machine(std::shared_ptr<Features> data)
{
	//certain setup params
	mkl_converged=false;
	verbosity=1 ;
	init_margin=0.15;
	init_iter=500;
	precision_violations=0;
	opt_precision=DEF_PRECISION;

	strcpy (learn_parm->predfile, "");
	learn_parm->biased_hyperplane=0;
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=get_qpsize();
	learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize-1;
	learn_parm->maxiter=100000;
	learn_parm->svm_iter_to_shrink=100;
	learn_parm->svm_c=C1;
	learn_parm->transduction_posratio=0.33;
	learn_parm->svm_costratio=C2/C1;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=epsilon; // GU: better decrease it ... ??
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;

    if (!kernel)
        error("SVM_light can not proceed without kernel!");

	if (data)
		kernel->init(data, data);

    if (!kernel->has_features())
        error("SVM_light can not proceed without initialized kernel!");

	int32_t num_vec=kernel->get_num_vec_lhs();
	io::info("num_vec={}", num_vec);


	m_labels=std::make_shared<BinaryLabels>(num_vec);

	std::static_pointer_cast<BinaryLabels>(m_labels)->set_to_one();

	// in case of LINADD enabled kernels cleanup!
	if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
		kernel->clear_normal() ;

	// output some info
	SG_DEBUG("threads = {}", env()->get_num_threads())
	SG_DEBUG("qpsize = {}", learn_parm->svm_maxqpsize)
	SG_DEBUG("epsilon = %1.1e", learn_parm->epsilon_crit)
	SG_DEBUG("kernel->has_property(KP_LINADD) = {}", kernel->has_property(KP_LINADD))
	SG_DEBUG("kernel->has_property(KP_KERNCOMBINATION) = {}", kernel->has_property(KP_KERNCOMBINATION))
	SG_DEBUG("kernel->has_property(KP_BATCHEVALUATION) = {}", kernel->has_property(KP_BATCHEVALUATION))
	SG_DEBUG("kernel->get_optimization_type() = {}", kernel->get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" : "SLOWBUTMEMEFFICIENT" )
	SG_DEBUG("get_solver_type() = {}", get_solver_type())
	SG_DEBUG("get_linadd_enabled() = {}", get_linadd_enabled())
	SG_DEBUG("get_batch_computation_enabled() = {}", get_batch_computation_enabled())
	SG_DEBUG("kernel->get_num_subkernels() = {}", kernel->get_num_subkernels())

	use_kernel_cache = !((kernel->get_kernel_type() == K_CUSTOM) ||
						 (get_linadd_enabled() && kernel->has_property(KP_LINADD)));

	SG_DEBUG("use_kernel_cache = {}", use_kernel_cache)

	if (kernel->get_kernel_type() == K_COMBINED)
	{
		for (index_t k_idx=0; k_idx<(std::static_pointer_cast<CombinedKernel>(kernel))->get_num_kernels(); k_idx++)
		{
			auto kn =  (std::static_pointer_cast<CombinedKernel>(kernel))->get_kernel(k_idx);
			// allocate kernel cache but clean up beforehand
			kn->resize_kernel_cache(kn->get_cache_size());

		}
	}

	kernel->resize_kernel_cache(kernel->get_cache_size());

	// train the svm
	svm_learn();

	// brain damaged svm light work around
	create_new_model(model->sv_num-1);
	set_bias(-model->b);
	for (int32_t i=0; i<model->sv_num-1; i++)
	{
		set_alpha(i, model->alpha[i+1]);
		set_support_vector(i, model->supvec[i+1]);
	}

	// in case of LINADD enabled kernels cleanup!
	if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
	{
		kernel->clear_normal() ;
		kernel->delete_optimization() ;
	}

	if (use_kernel_cache)
		kernel->kernel_cache_cleanup();

	return true ;
}
#endif //USE_SVMLIGHT

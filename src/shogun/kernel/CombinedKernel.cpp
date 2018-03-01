/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evangelos Anagnostopoulos, Heiko Strathmann,
 *          Evan Shelhamer, Sergey Lisitsyn, Roman Votyakov, Jacob Walker,
 *          Wu Lin, Michele Mazzoni, Evgeniy Andreev, Viktor Gal
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parallel.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/CombinedFeatures.h>
#include <string.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CCombinedKernel::CCombinedKernel(int32_t size, bool asw)
: CKernel(size), append_subkernel_weights(asw)
{
	init();

	if (append_subkernel_weights)
		SG_INFO("(subkernel weights are appended)\n")

	SG_INFO("Combined kernel created (%p)\n", this)
}

CCombinedKernel::~CCombinedKernel()
{
	SG_FREE(subkernel_weights_buffer);
	subkernel_weights_buffer=NULL;

	cleanup();
	SG_UNREF(kernel_array);

	SG_INFO("Combined kernel deleted (%p).\n", this)
}

void CCombinedKernel::init_subkernel_weights()
{
	weight_update=true;
	ASSERT(subkernel_log_weights.vlen>0);
	SGVector<float64_t> wt(subkernel_log_weights.vlen);

	Map<VectorXd> eigen_wt(wt.vector, wt.vlen);
	Map<VectorXd> eigen_log_wt(subkernel_log_weights.vector, subkernel_log_weights.vlen);

	// log_sum_exp trick
	float64_t max_coeff=eigen_log_wt.maxCoeff();
	VectorXd tmp = eigen_log_wt.array() - max_coeff;
	float64_t sum = std::log(tmp.array().exp().sum());
	eigen_wt = tmp.array() - sum;
	eigen_wt = eigen_wt.array().exp();
	set_subkernel_weights(wt);
}
bool CCombinedKernel::init_with_extracted_subsets(
    CFeatures* l, CFeatures* r, SGVector<index_t> lhs_subset,
    SGVector<index_t> rhs_subset)
{

	auto l_combined = dynamic_cast<CCombinedFeatures*>(l);
	auto r_combined = dynamic_cast<CCombinedFeatures*>(r);

	if (!l_combined || !r_combined)
		SG_ERROR("Cast failed - unsupported features passed")

	CKernel::init(l, r);
	ASSERT(l->get_feature_type() == F_UNKNOWN)
	ASSERT(r->get_feature_type() == F_UNKNOWN)

	CFeatures* lf = NULL;
	CFeatures* rf = NULL;
	CKernel* k = NULL;

	bool result = true;
	index_t f_idx = 0;

	SG_DEBUG("Starting for loop for kernels\n")
	for (index_t k_idx = 0; k_idx < get_num_kernels() && result; k_idx++)
	{
		k = get_kernel(k_idx);

		if (!k)
			SG_ERROR("Kernel at position %d is NULL\n", k_idx);

		// skip over features - the custom kernel does not need any
		if (k->get_kernel_type() != K_CUSTOM)
		{
			if (l_combined->get_num_feature_obj() > f_idx &&
			    r_combined->get_num_feature_obj() > f_idx)
			{
				lf = l_combined->get_feature_obj(f_idx);
				rf = r_combined->get_feature_obj(f_idx);
			}

			f_idx++;
			if (!lf || !rf)
			{
				SG_UNREF(lf);
				SG_UNREF(rf);
				SG_UNREF(k);
				SG_ERROR(
				    "CombinedKernel: Number of features/kernels does not "
				    "match - bailing out\n")
			}

			SG_DEBUG("Initializing 0x%p - \"%s\"\n", this, k->get_name())
			result = k->init(lf, rf);
			SG_UNREF(lf);
			SG_UNREF(rf);

			if (!result)
				break;
		}
		else
		{
			SG_DEBUG(
			    "Initializing 0x%p - \"%s\" (skipping init, this is a CUSTOM "
			    "kernel)\n",
			    this, k->get_name())
			if (!k->has_features())
				SG_ERROR(
				    "No kernel matrix was assigned to this Custom kernel\n")

			auto k_custom = dynamic_cast<CCustomKernel*>(k);
			if (!k_custom)
				SG_ERROR("Dynamic cast to custom kernel failed")

			// clear all previous subsets
			k_custom->remove_all_row_subsets();
			// apply new subset
			k_custom->add_row_subset(lhs_subset);

			k_custom->remove_all_col_subsets();
			// apply new subset
			k_custom->add_col_subset(rhs_subset);

			if (k->get_num_vec_lhs() != num_lhs)
				SG_ERROR(
				    "Number of lhs-feature vectors (%d) not match with number "
				    "of rows (%d) of custom kernel\n",
				    num_lhs, k->get_num_vec_lhs())
			if (k->get_num_vec_rhs() != num_rhs)
				SG_ERROR(
				    "Number of rhs-feature vectors (%d) not match with number "
				    "of cols (%d) of custom kernel\n",
				    num_rhs, k->get_num_vec_rhs())
		}

		SG_UNREF(k);
	}

	if (!result)
	{
		SG_INFO("CombinedKernel: Initialising the following kernel failed\n")
		if (k)
		{
			k->list_kernel();
			SG_UNREF(k);
		}
		else
			SG_INFO("<NULL>\n")
		return false;
	}

	if (l_combined->get_num_feature_obj() <= 0 ||
	    l_combined->get_num_feature_obj() != r_combined->get_num_feature_obj())
		SG_ERROR(
		    "CombinedKernel: Number of features/kernels does not match - "
		    "bailing out\n")

	init_normalizer();
	initialized = true;
	return true;
}

bool CCombinedKernel::init(CFeatures* l, CFeatures* r)
{
	if (enable_subkernel_weight_opt && !weight_update)
	{
		init_subkernel_weights();
	}

	if (!l)
		SG_ERROR("LHS features are NULL");
	if (!r)
		SG_ERROR("RHS features are NULL");

	SGVector<index_t> lhs_subset;
	SGVector<index_t> rhs_subset;

	CCombinedFeatures* combined_l;
	CCombinedFeatures* combined_r;

	auto l_subset_stack = l->get_subset_stack();
	auto r_subset_stack = r->get_subset_stack();

	if (l_subset_stack->has_subsets())
	{
		lhs_subset = l_subset_stack->get_last_subset()->get_subset_idx();
	}
	else
	{
		lhs_subset = SGVector<index_t>(l->get_num_vectors());
		lhs_subset.range_fill();
	}

	if (r_subset_stack->has_subsets())
	{
		rhs_subset = r_subset_stack->get_last_subset()->get_subset_idx();
	}
	else
	{
		rhs_subset = SGVector<index_t>(r->get_num_vectors());
		rhs_subset.range_fill();
	}

	SG_UNREF(l_subset_stack);
	SG_UNREF(r_subset_stack);

	/* if the specified features are not combined features, but a single other
	 * feature type, assume that the caller wants to use all kernels on these */
	if (l && r && l->get_feature_class() == r->get_feature_class() &&
	    l->get_feature_type() == r->get_feature_type() &&
	    l->get_feature_class() != C_COMBINED)
	{
		SG_DEBUG(
		    "Initialising combined kernel's combined features with the "
		    "same instance from parameters\n");
		/* construct combined features with each element being the parameter
		 * The we must make sure that we make any custom kernels aware of any
		 * subsets present!
		 */
		combined_l = new CCombinedFeatures();
		combined_r = new CCombinedFeatures();

		for (index_t i = 0; i < get_num_subkernels(); ++i)
		{
			combined_l->append_feature_obj(l);
			combined_r->append_feature_obj(r);
		}
	}
	else
	{
		combined_l = (CCombinedFeatures*)l;
		combined_r = (CCombinedFeatures*)r;
	}

	return init_with_extracted_subsets(
	    combined_l, combined_r, lhs_subset, rhs_subset);
}

void CCombinedKernel::remove_lhs()
{
	delete_optimization();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs();

		SG_UNREF(k);
	}
	CKernel::remove_lhs();

	num_lhs=0;
}

void CCombinedKernel::remove_rhs()
{
	delete_optimization();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_rhs();

		SG_UNREF(k);
	}
	CKernel::remove_rhs();

	num_rhs=0;
}

void CCombinedKernel::remove_lhs_and_rhs()
{
	delete_optimization();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs_and_rhs();

		SG_UNREF(k);
	}

	CKernel::remove_lhs_and_rhs();

	num_lhs=0;
	num_rhs=0;
}

void CCombinedKernel::cleanup()
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		k->cleanup();
		SG_UNREF(k);
	}

	delete_optimization();

	CKernel::cleanup();

	num_lhs=0;
	num_rhs=0;
}

void CCombinedKernel::list_kernels()
{
	SG_INFO("BEGIN COMBINED KERNEL LIST - ")
	this->list_kernel();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		k->list_kernel();
		SG_UNREF(k);
	}
	SG_INFO("END COMBINED KERNEL LIST - ")
}

float64_t CCombinedKernel::compute(int32_t x, int32_t y)
{
	float64_t result=0;
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		if (k->get_combined_kernel_weight()!=0)
			result += k->get_combined_kernel_weight() * k->kernel(x,y);
		SG_UNREF(k);
	}

	return result;
}

bool CCombinedKernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t *weights)
{
	SG_DEBUG("initializing CCombinedKernel optimization\n")

	delete_optimization();

	bool have_non_optimizable=false;

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);

		bool ret=true;

		if (k && k->has_property(KP_LINADD))
			ret=k->init_optimization(count, IDX, weights);
		else
		{
			SG_WARNING("non-optimizable kernel 0x%X in kernel-list\n", k)
			have_non_optimizable=true;
		}

		if (!ret)
		{
			have_non_optimizable=true;
			SG_WARNING("init_optimization of kernel 0x%X failed\n", k)
		}

		SG_UNREF(k);
	}

	if (have_non_optimizable)
	{
		SG_WARNING("some kernels in the kernel-list are not optimized\n")

		sv_idx=SG_MALLOC(int32_t, count);
		sv_weight=SG_MALLOC(float64_t, count);
		sv_count=count;
		for (int32_t i=0; i<count; i++)
		{
			sv_idx[i]=IDX[i];
			sv_weight[i]=weights[i];
		}
	}
	set_is_initialized(true);

	return true;
}

bool CCombinedKernel::delete_optimization()
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		if (k->has_property(KP_LINADD))
			k->delete_optimization();

		SG_UNREF(k);
	}

	SG_FREE(sv_idx);
	sv_idx = NULL;

	SG_FREE(sv_weight);
	sv_weight = NULL;

	sv_count = 0;
	set_is_initialized(false);

	return true;
}

void CCombinedKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* result, int32_t num_suppvec,
	int32_t* IDX, float64_t* weights, float64_t factor)
{
	ASSERT(num_vec<=get_num_vec_rhs())
	ASSERT(num_vec>0)
	ASSERT(vec_idx)
	ASSERT(result)

	//we have to do the optimization business ourselves but lets
	//make sure we start cleanly
	delete_optimization();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		if (k && k->has_property(KP_BATCHEVALUATION))
		{
			if (k->get_combined_kernel_weight()!=0)
				k->compute_batch(num_vec, vec_idx, result, num_suppvec, IDX, weights, k->get_combined_kernel_weight());
		}
		else
			emulate_compute_batch(k, num_vec, vec_idx, result, num_suppvec, IDX, weights);

		SG_UNREF(k);
	}

	//clean up
	delete_optimization();
}

void CCombinedKernel::emulate_compute_batch(
	CKernel* k, int32_t num_vec, int32_t* vec_idx, float64_t* result,
	int32_t num_suppvec, int32_t* IDX, float64_t* weights)
{
	ASSERT(k)
	ASSERT(result)

	if (k->has_property(KP_LINADD))
	{
		if (k->get_combined_kernel_weight()!=0)
		{
			k->init_optimization(num_suppvec, IDX, weights);

			#pragma omp parallel for
			for (int32_t i=0; i<num_vec; ++i)
				result[i] += k->get_combined_kernel_weight()*k->compute_optimized(vec_idx[i]);

			k->delete_optimization();
		}
	}
	else
	{
		ASSERT(IDX!=NULL || num_suppvec==0)
		ASSERT(weights!=NULL || num_suppvec==0)

		if (k->get_combined_kernel_weight()!=0)
		{ // compute the usual way for any non-optimized kernel
			#pragma omp parallel for
			for (int32_t i=0; i<num_vec; i++)
			{
				float64_t sub_result=0;
				for (int32_t j=0; j<num_suppvec; j++)
					sub_result += weights[j] * k->kernel(IDX[j], vec_idx[i]);

				result[i] += k->get_combined_kernel_weight()*sub_result;
			}
		}
	}
}

float64_t CCombinedKernel::compute_optimized(int32_t idx)
{
	if (!get_is_initialized())
	{
		SG_ERROR("CCombinedKernel optimization not initialized\n")
		return 0;
	}

	float64_t result=0;

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		if (k->has_property(KP_LINADD) &&
			k->get_is_initialized())
		{
			if (k->get_combined_kernel_weight()!=0)
			{
				result +=
					k->get_combined_kernel_weight()*k->compute_optimized(idx);
			}
		}
		else
		{
			ASSERT(sv_idx!=NULL || sv_count==0)
			ASSERT(sv_weight!=NULL || sv_count==0)

			if (k->get_combined_kernel_weight()!=0)
			{ // compute the usual way for any non-optimized kernel
				float64_t sub_result=0;
				for (int32_t j=0; j<sv_count; j++)
					sub_result += sv_weight[j] * k->kernel(sv_idx[j], idx);

				result += k->get_combined_kernel_weight()*sub_result;
			}
		}

		SG_UNREF(k);
	}

	return result;
}

void CCombinedKernel::add_to_normal(int32_t idx, float64_t weight)
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		k->add_to_normal(idx, weight);
		SG_UNREF(k);
	}
	set_is_initialized(true) ;
}

void CCombinedKernel::clear_normal()
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		k->clear_normal() ;
		SG_UNREF(k);
	}
	set_is_initialized(true) ;
}

void CCombinedKernel::compute_by_subkernel(
	int32_t idx, float64_t * subkernel_contrib)
{
	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			CKernel* k = get_kernel(k_idx);
			int32_t num = -1 ;
			k->get_subkernel_weights(num);
			if (num>1)
				k->compute_by_subkernel(idx, &subkernel_contrib[i]) ;
			else
				subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;

			SG_UNREF(k);
			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			CKernel* k = get_kernel(k_idx);
			if (k->get_combined_kernel_weight()!=0)
				subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;

			SG_UNREF(k);
			i++ ;
		}
	}
}

const float64_t* CCombinedKernel::get_subkernel_weights(int32_t& num_weights)
{
	SG_DEBUG("entering CCombinedKernel::get_subkernel_weights()\n")

	num_weights = get_num_subkernels() ;
	SG_FREE(subkernel_weights_buffer);
	subkernel_weights_buffer = SG_MALLOC(float64_t, num_weights);

	if (append_subkernel_weights)
	{
		SG_DEBUG("appending kernel weights\n")

		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			CKernel* k = get_kernel(k_idx);
			int32_t num = -1 ;
			const float64_t *w = k->get_subkernel_weights(num);
			ASSERT(num==k->get_num_subkernels())
			for (int32_t j=0; j<num; j++)
				subkernel_weights_buffer[i+j]=w[j] ;

			SG_UNREF(k);
			i += num ;
		}
	}
	else
	{
		SG_DEBUG("not appending kernel weights\n")
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			CKernel* k = get_kernel(k_idx);
			subkernel_weights_buffer[i] = k->get_combined_kernel_weight();

			SG_UNREF(k);
			i++ ;
		}
	}

	SG_DEBUG("leaving CCombinedKernel::get_subkernel_weights()\n")
	return subkernel_weights_buffer ;
}

SGVector<float64_t> CCombinedKernel::get_subkernel_weights()
{
	if (enable_subkernel_weight_opt && !weight_update)
	{
		ASSERT(subkernel_log_weights.vlen>0);
		init_subkernel_weights();
	}

	int32_t num=0;
	const float64_t* w=get_subkernel_weights(num);

	float64_t* weights = SG_MALLOC(float64_t, num);
	for (int32_t i=0; i<num; i++)
		weights[i] = w[i];


	return SGVector<float64_t>(weights, num);
}

void CCombinedKernel::set_subkernel_weights(SGVector<float64_t> weights)
{
	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			CKernel* k = get_kernel(k_idx);
			int32_t num = k->get_num_subkernels() ;
			ASSERT(i<weights.vlen)
			k->set_subkernel_weights(SGVector<float64_t>(&weights.vector[i],num, false));

			SG_UNREF(k);
			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			CKernel* k = get_kernel(k_idx);
			ASSERT(i<weights.vlen)
			k->set_combined_kernel_weight(weights.vector[i]);

			SG_UNREF(k);
			i++ ;
		}
	}
}

void CCombinedKernel::set_optimization_type(EOptimizationType t)
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		k->set_optimization_type(t);

		SG_UNREF(k);
	}

	CKernel::set_optimization_type(t);
}

bool CCombinedKernel::precompute_subkernels()
{
	if (get_num_kernels()==0)
		return false;

	CDynamicObjectArray* new_kernel_array = new CDynamicObjectArray();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		CKernel* k = get_kernel(k_idx);
		new_kernel_array->append_element(new CCustomKernel(k));

		SG_UNREF(k);
	}

	SG_UNREF(kernel_array);
	kernel_array=new_kernel_array;
	SG_REF(kernel_array);

	return true;
}

void CCombinedKernel::init()
{
	sv_count=0;
	sv_idx=NULL;
	sv_weight=NULL;
	subkernel_weights_buffer=NULL;
	initialized=false;

	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	kernel_array=new CDynamicObjectArray();
	SG_REF(kernel_array);

	SG_ADD((CSGObject**) &kernel_array, "kernel_array", "Array of kernels.",
	    MS_AVAILABLE);

	m_parameters->add_vector(&sv_idx, &sv_count, "sv_idx",
		 "Support vector index.");
	watch_param("sv_idx", &sv_idx, &sv_count);

	m_parameters->add_vector(&sv_weight, &sv_count, "sv_weight",
		 "Support vector weights.");
	watch_param("sv_weight", &sv_weight, &sv_count);

	SG_ADD(&append_subkernel_weights, "append_subkernel_weights",
	    "If subkernel weights are appended.", MS_AVAILABLE);
	SG_ADD(&initialized, "initialized", "Whether kernel is ready to be used.",
	    MS_NOT_AVAILABLE);

	enable_subkernel_weight_opt=false;
	subkernel_log_weights = SGVector<float64_t>(1);
	subkernel_log_weights[0] = 0;
	SG_ADD(&subkernel_log_weights, "subkernel_log_weights",
	    "subkernel weights", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&enable_subkernel_weight_opt, "enable_subkernel_weight_opt",
	    "enable subkernel weight opt", MS_NOT_AVAILABLE);

	weight_update = false;
	SG_ADD(&weight_update, "weight_update",
	    "weight update", MS_NOT_AVAILABLE);
}

void CCombinedKernel::enable_subkernel_weight_learning()
{
	weight_update = false;
	enable_subkernel_weight_opt=false;
	subkernel_log_weights=get_subkernel_weights();
	enable_subkernel_weight_opt=true;
	ASSERT(subkernel_log_weights.vlen>0);
	for(index_t idx=0; idx<subkernel_log_weights.vlen; idx++)
	{
		ASSERT(subkernel_log_weights[idx]>0);//weight should be positive
		subkernel_log_weights[idx] =
		    std::log(subkernel_log_weights[idx]); // in log domain
	}
}

SGMatrix<float64_t> CCombinedKernel::get_parameter_gradient(
		const TParameter* param, index_t index)
{
	SGMatrix<float64_t> result;

	if (!strcmp(param->m_name, "combined_kernel_weight"))
	{
		if (append_subkernel_weights)
		{
			for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
			{
				CKernel* k=get_kernel(k_idx);
				result=k->get_parameter_gradient(param, index);

				SG_UNREF(k);

				if (result.num_cols*result.num_rows>0)
					return result;
			}
		}
		else
		{
			for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
			{
				CKernel* k=get_kernel(k_idx);
				result=k->get_kernel_matrix();

				SG_UNREF(k);

				return result;
			}
		}
	}
	else
	{
		if (!strcmp(param->m_name, "subkernel_log_weights"))
		{
			if(enable_subkernel_weight_opt)
			{
				ASSERT(index>=0 && index<subkernel_log_weights.vlen);
				CKernel* k=get_kernel(index);
				result=k->get_kernel_matrix();
				SG_UNREF(k);
				if (weight_update)
					weight_update = false;
				float64_t factor = 1.0;
				Map<VectorXd> eigen_log_wt(subkernel_log_weights.vector, subkernel_log_weights.vlen);
				// log_sum_exp trick
				float64_t max_coeff = eigen_log_wt.maxCoeff();
				VectorXd tmp = eigen_log_wt.array() - max_coeff;
				float64_t log_sum = std::log(tmp.array().exp().sum());

				factor = subkernel_log_weights[index] - max_coeff - log_sum;
				factor = CMath::exp(factor) - CMath::exp(factor*2.0);

				Map<MatrixXd> eigen_res(result.matrix, result.num_rows, result.num_cols);
				eigen_res = eigen_res * factor;
			}
			else
			{
				CKernel* k=get_kernel(0);
				result=k->get_kernel_matrix();
				SG_UNREF(k);
				result.zero();
			}
			return result;
		}
		else
		{
			float64_t coeff;
			for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
			{
				CKernel* k=get_kernel(k_idx);
				SGMatrix<float64_t> derivative=
					k->get_parameter_gradient(param, index);

				coeff=1.0;

				if (!append_subkernel_weights)
					coeff=k->get_combined_kernel_weight();

				for (index_t g=0; g<derivative.num_rows; g++)
				{
					for (index_t h=0; h<derivative.num_cols; h++)
						derivative(g,h)*=coeff;
				}

				if (derivative.num_cols*derivative.num_rows>0)
				{
					if (result.num_cols==0 && result.num_rows==0)
						result=derivative;
					else
					{
						for (index_t g=0; g<derivative.num_rows; g++)
						{
							for (index_t h=0; h<derivative.num_cols; h++)
								result(g,h)+=derivative(g,h);
						}
					}
				}

				SG_UNREF(k);
			}
		}
	}

	return result;
}

CCombinedKernel* CCombinedKernel::obtain_from_generic(CKernel* kernel)
{
	if (kernel->get_kernel_type()!=K_COMBINED)
	{
		SG_SERROR("CCombinedKernel::obtain_from_generic(): provided kernel is "
				"not of type CCombinedKernel!\n");
	}

	/* since an additional reference is returned */
	SG_REF(kernel);
	return (CCombinedKernel*)kernel;
}

CList* CCombinedKernel::combine_kernels(CList* kernel_list)
{
	CList* return_list = new CList(true);
	SG_REF(return_list);

	if (!kernel_list)
		return return_list;

	if (kernel_list->get_num_elements()==0)
		return return_list;

	int32_t num_combinations = 1;
	int32_t list_index = 0;

	/* calculation of total combinations */
	CSGObject* list = kernel_list->get_first_element();
	while (list)
	{
		CList* c_list= dynamic_cast<CList* >(list);
		if (!c_list)
		{
			SG_SERROR("CCombinedKernel::combine_kernels() : Failed to cast list of type "
					"%s to type CList\n", list->get_name());
		}

		if (c_list->get_num_elements()==0)
		{
			SG_SERROR("CCombinedKernel::combine_kernels() : Sub-list in position %d "
					"is empty.\n", list_index);
		}

		num_combinations *= c_list->get_num_elements();

		if (kernel_list->get_delete_data())
			SG_UNREF(list);

		list = kernel_list->get_next_element();
		++list_index;
	}

	/* creation of CCombinedKernels */
	CDynamicObjectArray kernel_array(num_combinations);
	for (index_t i=0; i<num_combinations; ++i)
	{
		CCombinedKernel* c_kernel = new CCombinedKernel();
		return_list->append_element(c_kernel);
		kernel_array.push_back(c_kernel);
	}

	/* first pass */
	list = kernel_list->get_first_element();
	CList* c_list = dynamic_cast<CList* >(list);

	/* kernel index in the list */
	index_t kernel_index = 0;

	/* here we duplicate the first list in the following form
	*  a,b,c,d,   a,b,c,d  ......   a,b,c,d  ---- for  a total of num_combinations elements
	*/
	EKernelType prev_kernel_type = K_UNKNOWN;
	bool first_kernel = true;
	for (CSGObject* kernel=c_list->get_first_element(); kernel; kernel=c_list->get_next_element())
	{
		CKernel* c_kernel = dynamic_cast<CKernel* >(kernel);

		if (first_kernel)
			 first_kernel = false;
		else if (c_kernel->get_kernel_type()!=prev_kernel_type)
		{
			SG_SERROR("CCombinedKernel::combine_kernels() : Sub-list in position "
					"0 contains different types of kernels\n");
		}

		prev_kernel_type = c_kernel->get_kernel_type();

		for (index_t index=kernel_index; index<num_combinations; index+=c_list->get_num_elements())
		{
			CCombinedKernel* comb_kernel =
					dynamic_cast<CCombinedKernel* >(kernel_array.get_element(index));
			comb_kernel->append_kernel(c_kernel);
			SG_UNREF(comb_kernel);
		}
		++kernel_index;
		if (c_list->get_delete_data())
			SG_UNREF(kernel);
	}

	if (kernel_list->get_delete_data())
		SG_UNREF(list);

	/* how often each kernel of the sub-list must appear */
	int32_t freq = c_list->get_num_elements();

	/* in this loop we replicate each kernel freq times
	*  until we assign to all the CombinedKernels a sub-kernel from this list
	*  That is for num_combinations */
	list = kernel_list->get_next_element();
	list_index = 1;
	while (list)
	{
		c_list = dynamic_cast<CList* >(list);

		/* index of kernel in the list */
		kernel_index = 0;
		first_kernel = true;
		for (CSGObject* kernel=c_list->get_first_element(); kernel; kernel=c_list->get_next_element())
		{
			CKernel* c_kernel = dynamic_cast<CKernel* >(kernel);

			if (first_kernel)
				first_kernel = false;
			else if (c_kernel->get_kernel_type()!=prev_kernel_type)
			{
				SG_SERROR("CCombinedKernel::combine_kernels() : Sub-list in position "
						"%d contains different types of kernels\n", list_index);
			}

			prev_kernel_type = c_kernel->get_kernel_type();

			/* moves the index so that we keep filling in, the way we do, until we reach the end of the list of combinedkernels */
			for (index_t base=kernel_index*freq; base<num_combinations; base+=c_list->get_num_elements()*freq)
			{
				/* inserts freq consecutives times the current kernel */
				for (index_t index=0; index<freq; ++index)
				{
					CCombinedKernel* comb_kernel =
							dynamic_cast<CCombinedKernel* >(kernel_array.get_element(base+index));
					comb_kernel->append_kernel(c_kernel);
					SG_UNREF(comb_kernel);
				}
			}
			++kernel_index;

			if (c_list->get_delete_data())
				SG_UNREF(kernel);
		}

		freq *= c_list->get_num_elements();
		if (kernel_list->get_delete_data())
			SG_UNREF(list);
		list = kernel_list->get_next_element();
		++list_index;
	}

	return return_list;
}

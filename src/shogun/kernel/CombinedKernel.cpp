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
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/CombinedFeatures.h>
#include <string.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CombinedKernel::CombinedKernel() : Kernel()
{
	init();
}

CombinedKernel::CombinedKernel(int32_t size, bool asw)
: Kernel(size), append_subkernel_weights(asw)
{
	init();
}

CombinedKernel::~CombinedKernel()
{
	SG_FREE(subkernel_weights_buffer);
	subkernel_weights_buffer=NULL;

	cleanup();

}

void CombinedKernel::init_subkernel_weights()
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
bool CombinedKernel::init_with_extracted_subsets(
    std::shared_ptr<Features> l, std::shared_ptr<Features> r, SGVector<index_t> lhs_subset,
    SGVector<index_t> rhs_subset)
{

	auto l_combined = std::dynamic_pointer_cast<CombinedFeatures>(l);
	auto r_combined = std::dynamic_pointer_cast<CombinedFeatures>(r);

	if (!l_combined || !r_combined)
		error("Cast failed - unsupported features passed");

	Kernel::init(l, r);
	ASSERT(l->get_feature_type() == F_UNKNOWN)
	ASSERT(r->get_feature_type() == F_UNKNOWN)

	std::shared_ptr<Features> lf = NULL;
	std::shared_ptr<Features> rf = NULL;
	std::shared_ptr<Kernel> k = NULL;

	bool result = true;
	index_t f_idx = 0;

	SG_TRACE("Starting for loop for kernels");
	for (index_t k_idx = 0; k_idx < get_num_kernels() && result; k_idx++)
	{
		k = get_kernel(k_idx);

		if (!k)
			error("Kernel at position {} is NULL", k_idx);

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
				error(
				    "CombinedKernel: Number of features/kernels does not "
				    "match - bailing out");
			}

			SG_DEBUG("Initializing 0x{} - \"{}\"", fmt::ptr(this), k->get_name())
			result = k->init(lf, rf);



			if (!result)
				break;
		}
		else
		{
			SG_DEBUG(
			    "Initializing 0x{} - \"{}\" (skipping init, this is a CUSTOM "
			    "kernel)",
			    fmt::ptr(this), k->get_name())
			if (!k->has_features())
				error(
				    "No kernel matrix was assigned to this Custom kernel");

			auto k_custom = std::dynamic_pointer_cast<CustomKernel>(k);
			if (!k_custom)
				error("Dynamic cast to custom kernel failed");

			// clear all previous subsets
			k_custom->remove_all_row_subsets();
			// apply new subset
			k_custom->add_row_subset(lhs_subset);

			k_custom->remove_all_col_subsets();
			// apply new subset
			k_custom->add_col_subset(rhs_subset);

			if (k->get_num_vec_lhs() != num_lhs)
				error(
				    "Number of lhs-feature vectors ({}) not match with number "
				    "of rows ({}) of custom kernel",
				    num_lhs, k->get_num_vec_lhs());
			if (k->get_num_vec_rhs() != num_rhs)
				error(
				    "Number of rhs-feature vectors ({}) not match with number "
				    "of cols ({}) of custom kernel",
				    num_rhs, k->get_num_vec_rhs());
		}


	}

	if (!result)
	{
		io::info("CombinedKernel: Initialising the following kernel failed");
		if (k)
		{
			k->list_kernel();

		}
		else
			io::info("<NULL>");
		return false;
	}

	if (l_combined->get_num_feature_obj() <= 0 ||
	    l_combined->get_num_feature_obj() != r_combined->get_num_feature_obj())
		error(
		    "CombinedKernel: Number of features/kernels does not match - "
		    "bailing out");

	init_normalizer();
	initialized = true;
	return true;
}

bool CombinedKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	if (enable_subkernel_weight_opt && !weight_update)
	{
		init_subkernel_weights();
	}

	if (!l)
		error("LHS features are NULL");
	if (!r)
		error("RHS features are NULL");

	SGVector<index_t> lhs_subset;
	SGVector<index_t> rhs_subset;

	std::shared_ptr<CombinedFeatures> combined_l;
	std::shared_ptr<CombinedFeatures> combined_r;

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




	/* if the specified features are not combined features, but a single other
	 * feature type, assume that the caller wants to use all kernels on these */
	if (l && r && l->get_feature_class() == r->get_feature_class() &&
	    l->get_feature_type() == r->get_feature_type() &&
	    l->get_feature_class() != C_COMBINED)
	{
		SG_DEBUG(
		    "Initialising combined kernel's combined features with the "
		    "same instance from parameters");
		/* construct combined features with each element being the parameter
		 * The we must make sure that we make any custom kernels aware of any
		 * subsets present!
		 */
		combined_l = std::make_shared<CombinedFeatures>();
		combined_r = std::make_shared<CombinedFeatures>();

		for (index_t i = 0; i < get_num_subkernels(); ++i)
		{
			combined_l->append_feature_obj(l);
			combined_r->append_feature_obj(r);
		}
	}
	else
	{
		combined_l = std::static_pointer_cast<CombinedFeatures>(l);
		combined_r = std::static_pointer_cast<CombinedFeatures>(r);
	}

	return init_with_extracted_subsets(
	    combined_l, combined_r, lhs_subset, rhs_subset);
}

void CombinedKernel::remove_lhs()
{
	delete_optimization();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs();


	}
	Kernel::remove_lhs();

	num_lhs=0;
}

void CombinedKernel::remove_rhs()
{
	delete_optimization();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_rhs();


	}
	Kernel::remove_rhs();

	num_rhs=0;
}

void CombinedKernel::remove_lhs_and_rhs()
{
	delete_optimization();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs_and_rhs();


	}

	Kernel::remove_lhs_and_rhs();

	num_lhs=0;
	num_rhs=0;
}

void CombinedKernel::cleanup()
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		k->cleanup();

	}

	delete_optimization();

	Kernel::cleanup();

	num_lhs=0;
	num_rhs=0;
}

void CombinedKernel::list_kernels()
{
	io::info("BEGIN COMBINED KERNEL LIST - ");
	this->list_kernel();

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		k->list_kernel();

	}
	io::info("END COMBINED KERNEL LIST - ");
}

float64_t CombinedKernel::compute(int32_t x, int32_t y)
{
	float64_t result=0;
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		if (k->get_combined_kernel_weight()!=0)
			result += k->get_combined_kernel_weight() * k->kernel(x,y);

	}

	return result;
}

bool CombinedKernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t *weights)
{
	SG_TRACE("initializing CombinedKernel optimization");

	delete_optimization();

	bool have_non_optimizable=false;

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);

		bool ret=true;

		if (k && k->has_property(KP_LINADD))
			ret=k->init_optimization(count, IDX, weights);
		else
		{
			io::warn("non-optimizable kernel 0x%X in kernel-list", fmt::ptr(k.get()));
			have_non_optimizable=true;
		}

		if (!ret)
		{
			have_non_optimizable=true;
			io::warn("init_optimization of kernel 0x%X failed", fmt::ptr(k.get()));
		}


	}

	if (have_non_optimizable)
	{
		io::warn("some kernels in the kernel-list are not optimized");

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

bool CombinedKernel::delete_optimization()
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		if (k->has_property(KP_LINADD))
			k->delete_optimization();


	}

	SG_FREE(sv_idx);
	sv_idx = NULL;

	SG_FREE(sv_weight);
	sv_weight = NULL;

	sv_count = 0;
	set_is_initialized(false);

	return true;
}

void CombinedKernel::compute_batch(
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
		auto k = get_kernel(k_idx);
		if (k && k->has_property(KP_BATCHEVALUATION))
		{
			if (k->get_combined_kernel_weight()!=0)
				k->compute_batch(num_vec, vec_idx, result, num_suppvec, IDX, weights, k->get_combined_kernel_weight());
		}
		else
			emulate_compute_batch(k, num_vec, vec_idx, result, num_suppvec, IDX, weights);


	}

	//clean up
	delete_optimization();
}

void CombinedKernel::emulate_compute_batch(
	std::shared_ptr<Kernel> k, int32_t num_vec, int32_t* vec_idx, float64_t* result,
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

float64_t CombinedKernel::compute_optimized(int32_t idx)
{
	if (!get_is_initialized())
	{
		error("CombinedKernel optimization not initialized");
		return 0;
	}

	float64_t result=0;

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
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


	}

	return result;
}

void CombinedKernel::add_to_normal(int32_t idx, float64_t weight)
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		k->add_to_normal(idx, weight);

	}
	set_is_initialized(true) ;
}

void CombinedKernel::clear_normal()
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		k->clear_normal() ;

	}
	set_is_initialized(true) ;
}

void CombinedKernel::compute_by_subkernel(
	int32_t idx, float64_t * subkernel_contrib)
{
	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			auto k = get_kernel(k_idx);
			int32_t num = -1 ;
			k->get_subkernel_weights(num);
			if (num>1)
				k->compute_by_subkernel(idx, &subkernel_contrib[i]) ;
			else
				subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;


			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			auto k = get_kernel(k_idx);
			if (k->get_combined_kernel_weight()!=0)
				subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;


			i++ ;
		}
	}
}

const float64_t* CombinedKernel::get_subkernel_weights(int32_t& num_weights)
{
	SG_TRACE("entering CombinedKernel::get_subkernel_weights()");

	num_weights = get_num_subkernels() ;
	SG_FREE(subkernel_weights_buffer);
	subkernel_weights_buffer = SG_MALLOC(float64_t, num_weights);

	if (append_subkernel_weights)
	{
		SG_DEBUG("appending kernel weights")

		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			auto k = get_kernel(k_idx);
			int32_t num = -1 ;
			const float64_t *w = k->get_subkernel_weights(num);
			ASSERT(num==k->get_num_subkernels())
			for (int32_t j=0; j<num; j++)
				subkernel_weights_buffer[i+j]=w[j] ;


			i += num ;
		}
	}
	else
	{
		SG_DEBUG("not appending kernel weights")
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); ++k_idx)
		{
			auto k = get_kernel(k_idx);
			subkernel_weights_buffer[i] = k->get_combined_kernel_weight();

			++i;
		}
	}

	SG_TRACE("leaving CombinedKernel::get_subkernel_weights()");
	return subkernel_weights_buffer ;
}

SGVector<float64_t> CombinedKernel::get_subkernel_weights()
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

void CombinedKernel::set_subkernel_weights(SGVector<float64_t> weights)
{
	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			auto k = get_kernel(k_idx);
			int32_t num = k->get_num_subkernels() ;
			ASSERT(i<weights.vlen)
			k->set_subkernel_weights(SGVector<float64_t>(&weights.vector[i],num, false));


			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
		{
			auto k = get_kernel(k_idx);
			ASSERT(i<weights.vlen)
			k->set_combined_kernel_weight(weights.vector[i]);


			i++ ;
		}
	}
}

void CombinedKernel::set_optimization_type(EOptimizationType t)
{
	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		k->set_optimization_type(t);


	}

	Kernel::set_optimization_type(t);
}

bool CombinedKernel::precompute_subkernels()
{
	if (get_num_kernels()==0)
		return false;

	std::vector<std::shared_ptr<Kernel>> new_kernel_array;
	new_kernel_array.reserve(get_num_kernels());

	for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
	{
		auto k = get_kernel(k_idx);
		new_kernel_array.push_back(std::make_shared<CustomKernel>(k));
	}


	kernel_array=new_kernel_array;


	return true;
}

void CombinedKernel::init()
{
	sv_count=0;
	sv_idx=NULL;
	sv_weight=NULL;
	subkernel_weights_buffer=NULL;
	initialized=false;
	append_subkernel_weights = false;

	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	kernel_array.clear();

	SG_ADD(&kernel_array, "kernel_array", "Array of kernels.", ParameterProperties::HYPER);

	/*m_parameters->add_vector(&sv_idx, &sv_count, "sv_idx",
		 "Support vector index.");*/
	watch_param("sv_idx", &sv_idx, &sv_count);

	/*m_parameters->add_vector(&sv_weight, &sv_count, "sv_weight",
		 "Support vector weights.");*/
	watch_param("sv_weight", &sv_weight, &sv_count);

	SG_ADD(&append_subkernel_weights, "append_subkernel_weights",
	    "If subkernel weights are appended.", ParameterProperties::HYPER);
	SG_ADD(&initialized, "initialized", "Whether kernel is ready to be used.");

	enable_subkernel_weight_opt=false;
	subkernel_log_weights = SGVector<float64_t>(1);
	subkernel_log_weights[0] = 0;
	SG_ADD(&subkernel_log_weights, "subkernel_log_weights",
	    "subkernel weights", ParameterProperties::HYPER | ParameterProperties::GRADIENT);
	SG_ADD(&enable_subkernel_weight_opt, "enable_subkernel_weight_opt",
	    "enable subkernel weight opt");

	weight_update = false;
	SG_ADD(&weight_update, "weight_update",
	    "weight update");
}

void CombinedKernel::enable_subkernel_weight_learning()
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

SGMatrix<float64_t> CombinedKernel::get_parameter_gradient(
		const TParameter* param, index_t index)
{
	SGMatrix<float64_t> result;

	if (!strcmp(param->m_name, "combined_kernel_weight"))
	{
		if (append_subkernel_weights)
		{
			for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
			{
				auto k=get_kernel(k_idx);
				result=k->get_parameter_gradient(param, index);



				if (result.num_cols*result.num_rows>0)
					return result;
			}
		}
		else
		{
			for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
			{
				auto k=get_kernel(k_idx);
				result=k->get_kernel_matrix();



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
				auto k=get_kernel(index);
				result=k->get_kernel_matrix();

				if (weight_update)
					weight_update = false;
				float64_t factor = 1.0;
				Map<VectorXd> eigen_log_wt(subkernel_log_weights.vector, subkernel_log_weights.vlen);
				// log_sum_exp trick
				float64_t max_coeff = eigen_log_wt.maxCoeff();
				VectorXd tmp = eigen_log_wt.array() - max_coeff;
				float64_t log_sum = std::log(tmp.array().exp().sum());

				factor = subkernel_log_weights[index] - max_coeff - log_sum;
				factor = std::exp(factor) - std::exp(factor * 2.0);

				Map<MatrixXd> eigen_res(result.matrix, result.num_rows, result.num_cols);
				eigen_res = eigen_res * factor;
			}
			else
			{
				auto k=get_kernel(0);
				result=k->get_kernel_matrix();

				result.zero();
			}
			return result;
		}
		else
		{
			float64_t coeff;
			for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
			{
				auto k=get_kernel(k_idx);
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


			}
		}
	}

	return result;
}

std::shared_ptr<CombinedKernel> CombinedKernel::obtain_from_generic(std::shared_ptr<Kernel> kernel)
{
	if (kernel->get_kernel_type()!=K_COMBINED)
	{
		error("CombinedKernel::obtain_from_generic(): provided kernel is "
				"not of type CombinedKernel!");
	}

	/* since an additional reference is returned */

	return std::static_pointer_cast<CombinedKernel>(kernel);
}

std::shared_ptr<List> CombinedKernel::combine_kernels(std::shared_ptr<List> kernel_list)
{
	auto return_list = std::make_shared<List>(true);


	if (!kernel_list)
		return return_list;

	if (kernel_list->get_num_elements()==0)
		return return_list;

	int32_t num_combinations = 1;
	int32_t list_index = 0;

	/* calculation of total combinations */
	auto list = kernel_list->get_first_element();
	while (list)
	{
		auto c_list= std::dynamic_pointer_cast<List>(list);
		if (!c_list)
		{
			error("CombinedKernel::combine_kernels() : Failed to cast list of type "
					"{} to type List", list->get_name());
		}

		if (c_list->get_num_elements()==0)
		{
			error("CombinedKernel::combine_kernels() : Sub-list in position {} "
					"is empty.", list_index);
		}

		num_combinations *= c_list->get_num_elements();

		if (kernel_list->get_delete_data())


		list = kernel_list->get_next_element();
		++list_index;
	}

	/* creation of CombinedKernels */
	std::vector<std::shared_ptr<Kernel>> kernel_array;
	kernel_array.reserve(num_combinations);
	for (index_t i=0; i<num_combinations; ++i)
	{
		auto c_kernel = std::make_shared<CombinedKernel>();
		return_list->append_element(c_kernel);
		kernel_array.push_back(c_kernel);
	}

	/* first pass */
	list = kernel_list->get_first_element();
	auto c_list = std::dynamic_pointer_cast<List>(list);

	/* kernel index in the list */
	index_t kernel_index = 0;

	/* here we duplicate the first list in the following form
	*  a,b,c,d,   a,b,c,d  ......   a,b,c,d  ---- for  a total of num_combinations elements
	*/
	EKernelType prev_kernel_type = K_UNKNOWN;
	bool first_kernel = true;
	for (auto kernel=c_list->get_first_element(); kernel; kernel=c_list->get_next_element())
	{
		auto c_kernel = std::dynamic_pointer_cast<Kernel>(kernel);

		if (first_kernel)
			 first_kernel = false;
		else if (c_kernel->get_kernel_type()!=prev_kernel_type)
		{
			error("CombinedKernel::combine_kernels() : Sub-list in position "
					"0 contains different types of kernels");
		}

		prev_kernel_type = c_kernel->get_kernel_type();

		for (index_t index=kernel_index; index<num_combinations; index+=c_list->get_num_elements())
		{
			auto comb_kernel =
			    std::dynamic_pointer_cast<CombinedKernel>(kernel_array[index]);
			comb_kernel->append_kernel(c_kernel);

		}
		++kernel_index;
	}

	/* how often each kernel of the sub-list must appear */
	int32_t freq = c_list->get_num_elements();

	/* in this loop we replicate each kernel freq times
	*  until we assign to all the CombinedKernels a sub-kernel from this list
	*  That is for num_combinations */
	list = kernel_list->get_next_element();
	list_index = 1;
	while (list)
	{
		c_list = std::dynamic_pointer_cast<List>(list);

		/* index of kernel in the list */
		kernel_index = 0;
		first_kernel = true;
		for (auto kernel=c_list->get_first_element(); kernel; kernel=c_list->get_next_element())
		{
			auto c_kernel = std::dynamic_pointer_cast<Kernel>(kernel);

			if (first_kernel)
				first_kernel = false;
			else if (c_kernel->get_kernel_type()!=prev_kernel_type)
			{
				error("CombinedKernel::combine_kernels() : Sub-list in position "
						"{} contains different types of kernels", list_index);
			}

			prev_kernel_type = c_kernel->get_kernel_type();

			/* moves the index so that we keep filling in, the way we do, until we reach the end of the list of combinedkernels */
			for (index_t base=kernel_index*freq; base<num_combinations; base+=c_list->get_num_elements()*freq)
			{
				/* inserts freq consecutives times the current kernel */
				for (index_t index=0; index<freq; ++index)
				{
					auto comb_kernel =
					    std::dynamic_pointer_cast<CombinedKernel>(
					        kernel_array[base + index]);
					comb_kernel->append_kernel(c_kernel);

				}
			}
			++kernel_index;

		}

		freq *= c_list->get_num_elements();

		list = kernel_list->get_next_element();
		++list_index;
	}

	return return_list;
}

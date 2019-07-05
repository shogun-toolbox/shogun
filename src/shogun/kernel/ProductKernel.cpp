/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Roman Votyakov, Soeren Sonnenburg,
 *          Evangelos Anagnostopoulos
 */

#include <shogun/kernel/ProductKernel.h>
#include <shogun/kernel/CustomKernel.h>

using namespace shogun;

ProductKernel::ProductKernel(int32_t size) : Kernel(size)
{
	init();

	io::info("Product kernel created ({})", fmt::ptr(this));
}

ProductKernel::~ProductKernel()
{
	cleanup();


	io::info("Product kernel deleted ({}).", fmt::ptr(this));
}

//Adapted from CombinedKernel
bool ProductKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Kernel::init(l,r);
	ASSERT(l->get_feature_class()==C_COMBINED)
	ASSERT(r->get_feature_class()==C_COMBINED)
	ASSERT(l->get_feature_type()==F_UNKNOWN)
	ASSERT(r->get_feature_type()==F_UNKNOWN)

	std::shared_ptr<Features> lf=NULL;
	std::shared_ptr<Features> rf=NULL;
	std::shared_ptr<Kernel> k=NULL;

	bool result=true;

	index_t f_idx=0;
	for (index_t k_idx=0; k_idx<get_num_subkernels() && result; k_idx++)
	{
		k=get_kernel(k_idx);
		if (!k)
			error("Kernel at position {} is NULL", k_idx);

		// skip over features - the custom kernel does not need any
		if (k->get_kernel_type() != K_CUSTOM)
		{
			lf=(std::static_pointer_cast<CombinedFeatures>(l))->get_feature_obj(f_idx);
			rf=(std::static_pointer_cast<CombinedFeatures>(r))->get_feature_obj(f_idx);
			f_idx++;
			if (!lf || !rf)
			{
				error("ProductKernel: Number of features/kernels does not match - bailing out");
			}

			SG_DEBUG("Initializing 0x{} - \"{}\"", fmt::ptr(this), k->get_name())
			result=k->init(lf,rf);




			if (!result)
				break;
		}
		else
		{
			SG_DEBUG("Initializing 0x{} - \"{}\" (skipping init, this is a CUSTOM kernel)", fmt::ptr(this), k->get_name())
			if (!k->has_features())
				error("No kernel matrix was assigned to this Custom kernel");
			if (k->get_num_vec_lhs() != num_lhs)
				error("Number of lhs-feature vectors ({}) not match with number of rows ({}) of custom kernel", num_lhs, k->get_num_vec_lhs());
			if (k->get_num_vec_rhs() != num_rhs)
				error("Number of rhs-feature vectors ({}) not match with number of cols ({}) of custom kernel", num_rhs, k->get_num_vec_rhs());
		}


	}

	if (!result)
	{
		io::info("ProductKernel: Initialising the following kernel failed");
		if (k)
		{
			k->list_kernel();

		}
		else
			io::info("<NULL>");
		return false;
	}

	if ( (f_idx!=(std::static_pointer_cast<CombinedFeatures>(l))->get_num_feature_obj()) ||
			(f_idx!=(std::static_pointer_cast<CombinedFeatures>(r))->get_num_feature_obj()) )
		error("ProductKernel: Number of features/kernels does not match - bailing out");

	initialized=true;
	return true;
}

//Adapted from CombinedKernel
void ProductKernel::remove_lhs()
{
	for (index_t k_idx=0; k_idx<get_num_subkernels(); k_idx++)
	{
		auto k=get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs();


	}
	Kernel::remove_lhs();

	num_lhs=0;
}

//Adapted from CombinedKernel
void ProductKernel::remove_rhs()
{
	for (index_t k_idx=0; k_idx<get_num_subkernels(); k_idx++)
	{
		auto k=get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_rhs();

	}
	Kernel::remove_rhs();

	num_rhs=0;
}

//Adapted from CombinedKernel
void ProductKernel::remove_lhs_and_rhs()
{
	for (index_t k_idx=0; k_idx<get_num_subkernels(); k_idx++)
	{
		auto k=get_kernel(k_idx);
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs_and_rhs();

	}

	Kernel::remove_lhs_and_rhs();

	num_lhs=0;
	num_rhs=0;
}

//Adapted from CombinedKernel
void ProductKernel::cleanup()
{
	for (index_t k_idx=0; k_idx<get_num_subkernels(); k_idx++)
	{
		auto k=get_kernel(k_idx);
		k->cleanup();

	}

	Kernel::cleanup();

	num_lhs=0;
	num_rhs=0;
}

//Adapted from CombinedKernel
void ProductKernel::list_kernels()
{
	io::info("BEGIN PRODUCT KERNEL LIST - ");
	this->list_kernel();

	for (index_t k_idx=0; k_idx<get_num_subkernels(); k_idx++)
	{
		auto k=get_kernel(k_idx);
		k->list_kernel();

	}
	io::info("END PRODUCT KERNEL LIST - ");
}

//Adapted from CombinedKernel
float64_t ProductKernel::compute(int32_t x, int32_t y)
{
	float64_t result=1;
	for (index_t k_idx=0; k_idx<get_num_subkernels(); k_idx++)
	{
		auto k=get_kernel(k_idx);
		result *= k->get_combined_kernel_weight() * k->kernel(x,y);

	}

	return result;
}

//Adapted from CombinedKernel
bool ProductKernel::precompute_subkernels()
{
	if (get_num_subkernels()==0)
		return false;

	std::vector<std::shared_ptr<Kernel>> new_kernel_array;
	new_kernel_array.reserve(get_num_subkernels());

	for (index_t k_idx=0; k_idx<get_num_subkernels(); k_idx++)
	{
		auto k=get_kernel(k_idx);
		new_kernel_array.push_back(std::make_shared<CustomKernel>(k));
	}


	kernel_array=new_kernel_array;


	return true;
}

void ProductKernel::init()
{
	initialized=false;

	properties=KP_NONE;
	kernel_array.clear();

	SG_ADD(
	    &kernel_array, "kernel_array", "Array of kernels",
	    ParameterProperties::HYPER);
	SG_ADD(&initialized, "initialized", "Whether kernel is ready to be used");
}

SGMatrix<float64_t> ProductKernel::get_parameter_gradient(
		const TParameter* param, index_t index)
{
	auto k=get_kernel(0);
	SGMatrix<float64_t> temp_kernel=k->get_kernel_matrix();


	bool found_derivative=false;

	for (index_t g=0; g<temp_kernel.num_rows; g++)
	{
		for (int h=0; h<temp_kernel.num_cols; h++)
			temp_kernel(g,h)=1.0;
	}

	for (index_t k_idx=0; k_idx<get_num_subkernels(); k_idx++)
	{
		k=get_kernel(k_idx);
		SGMatrix<float64_t> cur_matrix=k->get_kernel_matrix();
		SGMatrix<float64_t> derivative=
			k->get_parameter_gradient(param, index);

		if (derivative.num_cols*derivative.num_rows > 0)
		{
			found_derivative=true;
			for (index_t g=0; g<derivative.num_rows; g++)
			{
				for (index_t h=0; h<derivative.num_cols; h++)
					temp_kernel(g,h)*=derivative(g,h);
			}
		}
		else
		{
			for (index_t g=0; g<cur_matrix.num_rows; g++)
			{
				for (index_t h=0; h<cur_matrix.num_cols; h++)
					temp_kernel(g,h)*=cur_matrix(g,h);
			}
		}


	}

	if (found_derivative)
		return temp_kernel;
	else
		return SGMatrix<float64_t>();
}

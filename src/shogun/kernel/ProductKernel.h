/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Roman Votyakov, Soeren Sonnenburg, Yuyu Zhang, 
 *          Sergey Lisitsyn, Evangelos Anagnostopoulos
 */

#ifndef _PRODUCTKERNEL_H___
#define _PRODUCTKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>

#include <shogun/features/Features.h>
#include <shogun/features/CombinedFeatures.h>

namespace shogun
{
class Features;
class CombinedFeatures;

/** @brief The Product kernel is used to combine a number of kernels into a
 * single ProductKernel object by element multiplication.
 *
 * It keeps pointers to the multiplied sub-kernels \f$k_m({\bf x}, {\bf x'})\f$
 *
 * It is defined as:
 *
 * \f[
 * k_{product}({\bf x}, {\bf x'}) = \prod_{m=1}^M k_m({\bf x}, {\bf x'})
 * \f]
 */
class ProductKernel : public Kernel
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		ProductKernel(int32_t size=10);

		virtual ~ProductKernel();

		/** initialize kernel
		 *
		 * @param lhs features of left-hand side
		 * @param rhs features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> lhs, std::shared_ptr<Features> rhs);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type PRODUCT
		 */
		virtual EKernelType get_kernel_type() { return K_PRODUCT; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type UNKNOWN
		 */
		virtual EFeatureType get_feature_type() { return F_UNKNOWN; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class COMBINED
		 */
		virtual EFeatureClass get_feature_class() { return C_COMBINED; }

		/** return the kernel's name
		 *
		 * @return name Product
		 */
		virtual const char* get_name() const { return "ProductKernel"; }

		/** list kernels */
		void list_kernels();

		/** get kernel
		 *
		 * @param idx index of kernel
		 * @return kernel at index idx
		 */
		inline std::shared_ptr<Kernel> get_kernel(int32_t idx)
		{
			return kernel_array[idx];
		}

		/** insert kernel at position idx
		 * Idx must be less than get_num_subkernels()
		 *
		 * @param k kernel
		 * @param idx the position where to add the kernel
		 * @return if inserting was successful
		 */
		inline bool insert_kernel(std::shared_ptr<Kernel> k, int32_t idx)
		{
			ASSERT(k)
			REQUIRE(
			    idx >= 0 && idx < get_num_subkernels(),
			    "Index idx (%d) is out of range (0-%d)", idx,
			    get_num_subkernels());

			adjust_num_lhs_rhs_initialized(k);

			if (!(k->has_property(KP_LINADD)))
				unset_property(KP_LINADD);

			kernel_array.insert(kernel_array.begin() + idx, k);
			return true;
		}

		/** append kernel
		 *
		 * @param k kernel
		 * @return if appending was successful
		 */
		inline bool append_kernel(std::shared_ptr<Kernel> k)
		{
			ASSERT(k)
			adjust_num_lhs_rhs_initialized(k);

			if (!(k->has_property(KP_LINADD)))
				unset_property(KP_LINADD);

			int32_t n = get_num_subkernels();
			kernel_array.push_back(k);
			return n+1==get_num_subkernels();
		}

		/** delete kernel at position idx
		 *
		 * @param idx the index of the kernel to delete
		 * @return if deleting was successful
		 */
		inline bool delete_kernel(int32_t idx)
		{
			REQUIRE(
			    idx >= 0 && idx < kernel_array.size(),
			    "Index idx (%d) is out of range (0-%d)", idx,
			    kernel_array.size());

			kernel_array.erase(kernel_array.begin() + idx);
			return true;
		}

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		inline int32_t get_num_subkernels()
		{
			return kernel_array.size();
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual bool has_features()
		{
			return initialized;
		}

		/** remove lhs from kernel */
		virtual void remove_lhs();

		/** remove rhs from kernel */
		virtual void remove_rhs();

		/** remove lhs and rhs from kernel */
		virtual void remove_lhs_and_rhs();

		/** precompute all sub-kernels */
		bool precompute_subkernels();

		/** casts kernel to combined kernel
		 * @param n kernel to cast
		 */
		std::shared_ptr<ProductKernel> KernelToProductKernel(std::shared_ptr<Kernel> n)
		{
			return std::dynamic_pointer_cast<ProductKernel>(n);
		}

		/** return derivative with respect to specified parameter
		 *
		 * @param  param the parameter
		 * @param index the index of the element if parameter is a vector
		 *
		 * @return gradient with respect to parameter
		 */
		SGMatrix<float64_t> get_parameter_gradient(const TParameter* param,
				index_t index=-1);

		/** Get the Kernel array
		 *
		 * @return kernel array
		 */
		inline std::vector<std::shared_ptr<Kernel>> get_array()
		{
			
			return kernel_array;
		}

	protected:
		/** compute kernel function
		 *
		 * @param x x
		 * @param y y
		 * @return computed kernel function
		 */
		virtual float64_t compute(int32_t x, int32_t y);

		/** adjust the variables num_lhs, num_rhs and initialized
		 * based on the kernel to be appended/inserted
		 *
		 * @param k kernel
		 */
		inline void adjust_num_lhs_rhs_initialized(std::shared_ptr<Kernel> k)
		{
			ASSERT(k)

			if (k->get_num_vec_lhs())
			{
				if (num_lhs)
					ASSERT(num_lhs==k->get_num_vec_lhs())
				num_lhs=k->get_num_vec_lhs();

				if (!get_num_subkernels())
				{
					initialized=true;
#ifdef USE_SVMLIGHT
					cache_reset();
#endif //USE_SVMLIGHT
				}
			}
			else
				initialized=false;

			if (k->get_num_vec_rhs())
			{
				if (num_rhs)
					ASSERT(num_rhs==k->get_num_vec_rhs())
				num_rhs=k->get_num_vec_rhs();

				if (!get_num_subkernels())
				{
					initialized=true;
#ifdef USE_SVMLIGHT
					cache_reset();
#endif //USE_SVMLIGHT
				}
			}
			else
				initialized=false;
		}

	private:
		void init();

	protected:
		/** array of kernels */
		std::vector<std::shared_ptr<Kernel>> kernel_array;
		/** whether kernel is ready to be used */
		bool initialized;
};
}
#endif /* _PRODUCTKERNEL_H__ */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 * 
 * Code adapted from CCombinedKernel
 */

#ifndef _PRODUCTKERNEL_H___
#define _PRODUCTKERNEL_H___

#include <shogun/lib/List.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>

#include <shogun/features/Features.h>
#include <shogun/features/CombinedFeatures.h>

namespace shogun
{
class CFeatures;
class CCombinedFeatures;
class CList;
class CListElement;
/**
 * @brief The Product kernel is used to combine a number of kernels into a
 * single ProductKernel object by element multiplication.
 *
 * It keeps pointers to the multiplied sub-kernels \f$k_m({\bf x}, {\bf x'})\f$
 *
 *
 * It is defined as:
 *
 * \f[
 *     k_{product}({\bf x}, {\bf x'}) = \prod_{m=1}^M k_m({\bf x}, {\bf x'})
 * \f]
 *
 */
class CProductKernel : public CKernel
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * 
		 */
		CProductKernel(int32_t size=10);

		virtual ~CProductKernel();

		/** initialize kernel
		 *
		 * @param lhs features of left-hand side
		 * @param rhs features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* lhs, CFeatures* rhs);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type PRODUCT
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_PRODUCT;
		}

		/** return feature type the kernel can deal with
		 *
		 * @return feature type UNKNOWN
		 */
		virtual EFeatureType get_feature_type()
		{
			return F_UNKNOWN;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class COMBINED
		 */
		virtual EFeatureClass get_feature_class()
		{
			return C_COMBINED;
		}

		/** return the kernel's name
		 *
		 * @return name Product
		 */
		virtual const char* get_name() const { return "ProductKernel"; }

		/** list kernels */
		void list_kernels();

		/** get first kernel
		 *
		 * @return first kernel
		 */
		inline CKernel* get_first_kernel()
		{
			return (CKernel*) kernel_list->get_first_element();
		}

		/** get first kernel
		 *
		 * @param current
		 * @return first kernel
		 */
		inline CKernel* get_first_kernel(CListElement*& current)
		{
			return (CKernel*) kernel_list->get_first_element(current);
		}

		/** get kernel
		 *
		 * @param idx index of kernel
		 * @return kernel at index idx
		 */
		inline CKernel* get_kernel(int32_t idx)
		{
			CKernel * k = get_first_kernel();
			for (int32_t i=0; i<idx; i++)
			{
				SG_UNREF(k);
				k = get_next_kernel();
			}
			return k;
		}

		/** get last kernel
		 *
		 * @return last kernel
		 */
		inline CKernel* get_last_kernel()
		{
			return (CKernel*) kernel_list->get_last_element();
		}

		/** get next kernel
		 *
		 * @return next kernel
		 */
		inline CKernel* get_next_kernel()
		{
			return (CKernel*) kernel_list->get_next_element();
		}

		/** get next kernel multi-thread safe
		 *
		 * @param current
		 * @return next kernel
		 */
		inline CKernel* get_next_kernel(CListElement*& current)
		{
			return (CKernel*) kernel_list->get_next_element(current);
		}

		/** insert kernel
		 *
		 * @param k kernel
		 * @return if inserting was successful
		 */
		inline bool insert_kernel(CKernel* k)
		{
			ASSERT(k);
			adjust_num_lhs_rhs_initialized(k);

			if (!(k->has_property(KP_LINADD)))
				unset_property(KP_LINADD);

			return kernel_list->insert_element(k);
		}

		/** append kernel
		 *
		 * @param k kernel
		 * @return if appending was successful
		 */
		inline bool append_kernel(CKernel* k)
		{
			ASSERT(k);
			adjust_num_lhs_rhs_initialized(k);

			if (!(k->has_property(KP_LINADD)))
				unset_property(KP_LINADD);

			return kernel_list->append_element(k);
		}


		/** delete kernel
		 *
		 * @return if deleting was successful
		 */
		inline bool delete_kernel()
		{
			CKernel* k=(CKernel*) kernel_list->delete_element();
			SG_UNREF(k);

			if (!k)
			{
				num_lhs=0;
				num_rhs=0;
			}

			return (k!=NULL);
		}


		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		inline int32_t get_num_subkernels()
		{
		    return kernel_list->get_num_elements();
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual inline bool has_features()
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
		CProductKernel* KernelToProductKernel(shogun::CKernel* n)
		{
			return dynamic_cast<CProductKernel*>(n);
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
		inline void adjust_num_lhs_rhs_initialized(CKernel* k)
		{
			ASSERT(k);

			if (k->get_num_vec_lhs())
			{
				if (num_lhs)
					ASSERT(num_lhs==k->get_num_vec_lhs());
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
					ASSERT(num_rhs==k->get_num_vec_rhs());
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
		/** list of kernels */
		CList* kernel_list;
		/** whether kernel is ready to be used */
		bool initialized;
};
}
#endif /* _PRODUCTKERNEL_H__ */

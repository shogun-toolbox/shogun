#ifndef _COMBINEDKERNEL_H___
#define _COMBINEDKERNEL_H___

#include "lib/List.h"


///TODO
///guilib ->Combined stuff d.h. {add,del?}_{kernel/features}?#$*
///
class CCombinedKernel : public CKernel
{
	public:
		CCombinedKernel(LONG size);
		virtual ~CCombinedKernel();

		/** initialize kernel cache
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 *  set do_init to true if you want the kernel to call its setup function (like getting scaling parameters,...) */
		virtual bool init(CFeatures* lhs, CFeatures* rhs, bool do_init);

		/// clean up your kernel
		virtual void cleanup();

		/// load and save kernel init_data
		virtual bool load_init(FILE* src)
		{
			return false;
		}

		virtual bool save_init(FILE* dest)
		{
			return false;
		}
		
		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EKernelType get_kernel_type()
		{
			return K_COMBINED;
		}

		/** return feature type the kernel can deal with
		  */
		virtual EFeatureType get_feature_type()
		{
			return F_UNKNOWN;
		}

		/** return feature class the kernel can deal with
		  */
		virtual EFeatureClass get_feature_class()
		{
			return C_COMBINED;
		}

		// return the name of a kernel
		virtual const CHAR* get_name()
		{
			return "Combined";
		}

		void list_kernels();

		inline CKernel* get_first_kernel()
		{
			return kernel_list->get_first_element();
		}

		inline CKernel* get_next_kernel()
		{
			return kernel_list->get_next_element();
		}

		inline bool insert_kernel(CKernel* k)
		{
			return kernel_list->insert_element(k);
		}

		inline bool append_kernel(CKernel* k)
		{
			return kernel_list->append_element(k);
		}

		inline bool delete_kernel()
		{
			return kernel_list->delete_element();
		}

		inline int get_num_kernel()
		{
			return kernel_list->get_num_elements();
		}

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual REAL compute(INT x, INT y);

	protected:
		CList<CKernel*>* kernel_list;
};
#endif

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _COMBINEDKERNEL_H___
#define _COMBINEDKERNEL_H___

#include "lib/List.h"
#include "kernel/Kernel.h"
#include "lib/io.h"

class CCombinedKernel : public CKernel
{
	public:
		CCombinedKernel(LONG size, bool append_subkernel_weights);
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
		inline CKernel* get_first_kernel(CListElement<CKernel*>*&current)
		{
			return kernel_list->get_first_element(current);
		}

		inline CKernel* get_last_kernel()
		{
			return kernel_list->get_last_element();
		}

		inline CKernel* get_next_kernel(const CKernel* current)
		{
			ASSERT(kernel_list->get_current_element()==current) ;
			return kernel_list->get_next_element();
		}

		// multi-thread safe
		inline CKernel* get_next_kernel(CListElement<CKernel*> *&current)
		{
			return kernel_list->get_next_element(current);
		}

		inline bool insert_kernel(CKernel* k)
		{
			if (!(k->has_property(KP_LINADD)))
				unset_property(KP_LINADD);

			return kernel_list->insert_element(k);
		}

		inline bool append_kernel(CKernel* k)
		{
			if (!(k->has_property(KP_LINADD)))
				unset_property(KP_LINADD);

			return kernel_list->append_element(k);
		}

		inline bool delete_kernel()
		{
			return kernel_list->delete_element();
		}

		inline bool get_append_subkernel_weights()
			{
				return append_subkernel_weights ;
			}
		
		inline int get_num_subkernels()
		{
			if (append_subkernel_weights)
			{
				INT num_subkernels = 0 ;
				CListElement<CKernel*> *current = NULL ;
				CKernel * kn = get_first_kernel(current) ;
				while(kn)
				{
					num_subkernels += kn->get_num_subkernels() ;
					kn = get_next_kernel(current) ;
				}
				return num_subkernels ;
			}
			else
				return kernel_list->get_num_elements();
		}

		/// takes all necessary steps if the lhs is removed from kernel
		virtual void remove_lhs();
		/// takes all necessary steps if the rhs is removed from kernel
		virtual void remove_rhs();

		virtual bool init_optimization(INT count, INT *IDX, DREAL * weights);
		virtual bool delete_optimization();
		virtual DREAL compute_optimized(INT idx);
		virtual DREAL* compute_batch(INT& num_vec, DREAL* target, INT num_suppvec, INT* IDX, DREAL* weights, DREAL factor=1.0);
		/// emulates batch computation, via linadd optimization w^t x or even down to \sum_i alpha_i K(x_i,x)
		void emulate_compute_batch(CKernel* k, INT num_vec, DREAL* target, INT num_suppvec, INT* IDX, DREAL* weights);

		virtual void add_to_normal(INT idx, DREAL weight) ;
		virtual void clear_normal();
		virtual void compute_by_subkernel(INT idx, DREAL * subkernel_contrib);
		virtual const DREAL* get_subkernel_weights(INT& num_weights);
		virtual void set_subkernel_weights(DREAL* weights, INT num_weights);

		virtual void set_optimization_type(EOptimizationType t);

		virtual void set_precompute_matrix(bool flag, bool subkernel_flag) 
		{ 
			precompute_matrix = flag; 
			precompute_subkernel_matrix = subkernel_flag; 

			if (!precompute_matrix)
			{
				delete[] precomputed_matrix;
				precomputed_matrix = NULL;
			}
			CListElement<CKernel*> *current = NULL ;
			CKernel *kn = get_first_kernel(current);
			while (kn)
			{
				kn->set_precompute_matrix(subkernel_flag,false);
				kn = get_next_kernel(current);
			}
		}

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT x, INT y);

	protected:
		CList<CKernel*>* kernel_list;
		INT   sv_count;
		INT*  sv_idx;
		DREAL* sv_weight;
		DREAL* subkernel_weights_buffer;
		bool append_subkernel_weights;
};
#endif

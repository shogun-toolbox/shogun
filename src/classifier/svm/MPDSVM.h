/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MPDSVM_H___
#define _MPDSVM_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "lib/Cache.h"

/** class MPDSVM */
class CMPDSVM : public CSVM
{
	public:
		/** default constructor */
		CMPDSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CMPDSVM(DREAL C, CKernel* k, CLabels* lab);
		virtual ~CMPDSVM();

		/** train SVM */
		virtual bool train();

		/** get classifier type
		 *
		 * @return classifier type MPD
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_MPD; }

	protected:
		/** compute H
		 *
		 * @param i index of H
		 * @param j index of H
		 * @return computed H at index i,j
		 */
		inline DREAL compute_H(int i, int j)
		{
			return labels->get_label(i)*labels->get_label(j)*kernel->kernel(i,j);
		}

		/** lock kernel row
		 *
		 * @param i row to lock
		 * @return locked row
		 */
		inline KERNELCACHE_ELEM* lock_kernel_row(int i)
		{
			KERNELCACHE_ELEM* line=NULL;

			if (kernel_cache->is_cached(i))
			{
				line=kernel_cache->lock_entry(i);
				ASSERT(line);
			}

			if (!line)
			{
				line=kernel_cache->set_entry(i);
				ASSERT(line);
				CLabels* l=CKernelMachine::get_labels();
				ASSERT(l);

				for (int j=0; j<l->get_num_labels(); j++)
					line[j]=(KERNELCACHE_ELEM) l->get_label(i)*l->get_label(j)*kernel->kernel(i,j);
			}

			return line;
		}

		/** unlock kernel row
		 *
		 * @param i row to unlock
		 */
		inline void unlock_kernel_row(int i)
		{
			kernel_cache->unlock_entry(i);
		}

		/** kernel cache */
		CCache<KERNELCACHE_ELEM>* kernel_cache;
};

#endif  /* _MPDSVM_H___ */

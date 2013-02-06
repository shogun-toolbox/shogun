/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MPDSVM_H___
#define _MPDSVM_H___
#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/Cache.h>
#include <shogun/labels/BinaryLabels.h>

namespace shogun
{
/** @brief class MPDSVM */
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
		CMPDSVM(float64_t C, CKernel* k, CLabels* lab);
		virtual ~CMPDSVM();

		/** get classifier type
		 *
		 * @return classifier type MPD
		 */
		virtual EMachineType get_classifier_type() { return CT_MPD; }

		/** @return object name */
		virtual const char* get_name() const { return "MPDSVM"; }

	protected:
		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** compute H
		 *
		 * @param i index of H
		 * @param j index of H
		 * @return computed H at index i,j
		 */
		inline float64_t compute_H(int32_t i, int32_t j)
		{
			return ((CBinaryLabels*) m_labels)->get_label(i)*
				((CBinaryLabels*) m_labels)->get_label(j)*kernel->kernel(i,j);
		}

		/** lock kernel row
		 *
		 * @param i row to lock
		 * @return locked row
		 */
		inline KERNELCACHE_ELEM* lock_kernel_row(int32_t i)
		{
			KERNELCACHE_ELEM* line=NULL;

			if (kernel_cache->is_cached(i))
			{
				line=kernel_cache->lock_entry(i);
				ASSERT(line)
			}

			if (!line)
			{
				line=kernel_cache->set_entry(i);
				ASSERT(line)

				for (int32_t j=0; j<m_labels->get_num_labels(); j++)
					line[j]=(KERNELCACHE_ELEM) ((CBinaryLabels*) m_labels)->get_label(i)*((CBinaryLabels*) m_labels)->get_label(j)*kernel->kernel(i,j);
			}

			return line;
		}

		/** unlock kernel row
		 *
		 * @param i row to unlock
		 */
		inline void unlock_kernel_row(int32_t i)
		{
			kernel_cache->unlock_entry(i);
		}

		/** kernel cache */
		CCache<KERNELCACHE_ELEM>* kernel_cache;
};
}
#endif  /* _MPDSVM_H___ */

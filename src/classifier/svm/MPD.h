#ifndef _MPDSVM_H___
#define _MPDSVM_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "lib/Cache.h"

class CMPDSVM : public CSVM
{
	public:
		CMPDSVM();
		virtual ~CMPDSVM();
		virtual bool train();

		inline EClassifierType get_classifier_type() { return CT_MPD; }
	protected:
		inline DREAL CMPDSVM::compute_H(int i, int j)
		{
			CLabels* l = CKernelMachine::get_labels();
			return l->get_label(i)*l->get_label(j)*kernel->kernel(i,j);
		}

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
				CLabels* l = CKernelMachine::get_labels();
				ASSERT(line);
				ASSERT(l);

				for (int j=0; j<l->get_num_labels(); j++)
					line[j]=(KERNELCACHE_ELEM) l->get_label(i)*l->get_label(j)*kernel->kernel(i,j);
			}

			return line;
		}

		inline void unlock_kernel_row(int i)
		{
			kernel_cache->unlock_entry(i);
		}

		CCache<KERNELCACHE_ELEM>* kernel_cache;
};

#endif

#ifndef _KNN_H___
#define _KNN_H___

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "kernel/Kernel.h"
#include "kernel/KernelMachine.h"

class CKNN : public CKernelMachine
{
	public:
		CKNN();
		virtual ~CKNN();

		virtual bool train();
		virtual REAL* test();

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		inline void set_k(REAL k) 
		{
			assert(k>0);
			this->k=k;
		}

		inline REAL get_k()
		{
			return k;
		}

	protected:
		/// the k parameter in KNN
		REAL k;

		///	number of classes (i.e. number of values labels can take)
		int num_classes;

		///	smallest label, i.e. -1
		int min_label;

		/// number of train examples
		int num_train_labels;

		/// the actual trainlabels
		INT* train_labels;
};
#endif


#ifndef _KNN_H___
#define _KNN_H___

#include "lib/common.h"
#include "features/Features.h"
#include "kernel/Kernel.h"
#include "kernel/KernelMachine.h"

#include <stdio.h>

class CKNN : public CKernelMachine
{
	public:
		CKNN();
		virtual ~CKNN();

		bool load(FILE* svm_file);
		bool save(FILE* svm_file);

		inline void set_K(REAL k) {this->k=k; }

		REAL get_K() { return k; }

		// knn does not need training
		virtual bool train() { return true; }
		virtual REAL* test();

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* srcfile);

	protected:
		REAL k;
};
#endif


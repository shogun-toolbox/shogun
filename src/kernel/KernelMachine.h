#ifndef _KERNEL_MACHINE_H__
#define _KERNEL_MACHINE_H__

#include "lib/common.h"
#include "kernel/Kernel.h"
#include "features/Labels.h"

#include <stdio.h>

class CKernelMachine
{
	public:
		CKernelMachine();
		virtual ~CKernelMachine();

		virtual bool	train()=0;
		virtual REAL*	test()=0;
		virtual bool load(FILE* srcfile)=0;
		virtual bool save(FILE* dstfile)=0;

		virtual inline void set_labels(CLabels* lab) { labels=lab; }
		virtual CLabels* get_labels() { return labels; }

		inline void set_kernel(CKernel* k) { kernel=k; }
		inline CKernel* get_kernel() { return kernel; }

	protected:
		CKernel* kernel;
		CLabels* labels;
};
#endif

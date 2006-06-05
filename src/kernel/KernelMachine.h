#ifndef _KERNEL_MACHINE_H__
#define _KERNEL_MACHINE_H__

#include "lib/common.h"
#include "kernel/Kernel.h"
#include "features/Labels.h"
#include "classifier/Classifier.h"

#include <stdio.h>

class CKernelMachine : public CClassifier
{
	public:
		CKernelMachine();
		virtual ~CKernelMachine();

		inline void set_kernel(CKernel* k) { kernel=k; }
		inline CKernel* get_kernel() { return kernel; }

	protected:
		CKernel* kernel;
};
#endif

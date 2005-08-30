#ifndef _KERNELPERCEPTRON_H___
#define _KERNELPERCEPTRON_H___

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "kernel/KernelMachine.h"

class CKernelPerceptron : public CKernelMachine
{
	public:
		CKernelPerceptron();
		virtual ~CKernelPerceptron();

		virtual bool train();
		virtual REAL* test();

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);
};
#endif


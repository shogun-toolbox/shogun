#ifndef _PERCEPTRON_H___
#define _PERCEPTRON_H___

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "kernel/Kernel.h"
#include "kernel/KernelMachine.h"

class CPerceptron : public CKernelMachine
{
	public:
		CPerceptron();
		virtual ~CPerceptron();

		virtual bool train();
		virtual REAL* test();

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);
};
#endif


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

		virtual REAL classify_example(INT num);

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_KERNELPERCEPTRON;
		}
};
#endif


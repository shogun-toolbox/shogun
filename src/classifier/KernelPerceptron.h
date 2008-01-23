/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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

		virtual DREAL classify_example(INT num);

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_KERNELPERCEPTRON;
		}
};
#endif


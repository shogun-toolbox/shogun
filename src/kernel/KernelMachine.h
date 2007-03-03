/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNEL_MACHINE_H__
#define _KERNEL_MACHINE_H__

#include "lib/common.h"
#include "kernel/Kernel.h"
#include "features/Labels.h"
#include "classifier/Classifier.h"

#include <stdio.h>

class CClassifier;

class CKernelMachine : public CClassifier
{
	public:
		CKernelMachine();
		virtual ~CKernelMachine();

		inline void set_kernel(CKernel* k)
		{
            SG_REF(k);
			kernel=k;
		}

		inline CKernel* get_kernel()
		{
            SG_REF(kernel);
			return kernel;
		}

		virtual CLabels* classify(CLabels* output=NULL);

	protected:
		CKernel* kernel;
};
#endif

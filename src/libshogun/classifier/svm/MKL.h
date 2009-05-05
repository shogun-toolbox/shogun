/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#ifndef __MKL_H__
#define __MKL_H__

#include "lib/common.h"
#include "features/Features.h"
#include "kernel/Kernel.h"
#include "classifier/SVM.h"

class CMKL : public CSVM
{
	public:
		CMKL(CSVM* s=NULL) : svm(NULL)
		{
			set_constraint_generator(s);
		}
		CMKL(CSVM* s=NULL) : svm(NULL)
		{
			set_constraint_generator(s);
		}

		~CMKL()
		{
			SG_UNREF(svm);
		}

		void set_constraint_generator(CSVM* s)
		{
			SG_UNREF(svm);
			SG_REF(s);
			svm=s;
		}

	protected:
		CSVM* svm;
};

#endif //__MKL_H__

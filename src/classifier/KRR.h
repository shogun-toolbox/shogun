/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Mikio L. Braun
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KRR_H__
#define _KRR_H__

#include "lib/config.h"

#ifdef HAVE_LAPACK

#include "kernel/KernelMachine.h"

class CKRR : public CKernelMachine
{
	public:
		CKRR();
		virtual ~CKRR();

		/// set regularization constant
		inline void set_tau(DREAL t) { tau = t; };

		virtual bool train();

		virtual CLabels* classify(CLabels* output=NULL);
		virtual DREAL classify_example(INT num);

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_KRR;
		}

 private:
		DREAL *alpha;
		DREAL tau;
};

#endif /* HAVE_LAPACK */

#endif

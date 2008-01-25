/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVM_ONECLASS_H___
#define _LIBSVM_ONECLASS_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"

#include <stdio.h>

/** class LibSVMOneClass */
class CLibSVMOneClass : public CSVM
{
	public:
		/** default constructor */
		CLibSVMOneClass();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 */
		CLibSVMOneClass(DREAL C, CKernel* k);
		virtual ~CLibSVMOneClass();

		/** train SVM */
		virtual bool train();

		/** get classifier type
		 *
		 * @return classifier type LIBSVMONECLASS
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIBSVMONECLASS; }

	protected:
		/** SVM problem */
		svm_problem problem;
		/** SVM parameter */
		svm_parameter param;

		/** SVM model */
		struct svm_model* model;
};
#endif

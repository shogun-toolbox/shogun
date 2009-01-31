/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVM_H___
#define _LIBSVM_H___

#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"

#include <stdio.h>

/** LibSVM */
class CLibSVM : public CSVM
{
	public:
		/** constructor */
		CLibSVM();
		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CLibSVM(float64_t C, CKernel* k, CLabels* lab);

		virtual ~CLibSVM();

		/** train SVM */
		virtual bool train();

		/** get classifier type
		 *
		 * @return classifier type LIBSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIBSVM; }

		/** @return object name */
		inline virtual const char* get_name() { return "LibSVM"; }
	protected:
		/** SVM problem */
		svm_problem problem;
		/** SVM param */
		svm_parameter param;

		/** SVM model */
		struct svm_model* model;
};
#endif

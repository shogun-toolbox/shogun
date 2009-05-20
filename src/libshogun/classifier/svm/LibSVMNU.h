/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVMNU_H___
#define _LIBSVMNU_H___

#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"

#include <stdio.h>

/** @brief LibSVM */
class CLibSVMNu : public CSVM
{
	public:
		/** constructor */
		CLibSVMNu();
		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CLibSVMNu(float64_t C, CKernel* k, CLabels* lab);

		virtual ~CLibSVMNu();

		/** train SVM */
		virtual bool train();

		/** get classifier type
		 *
		 * @return classifier type LIBSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIBSVMNU; }

		/** @return object name */
		inline virtual const char* get_name() const { return "LibSVMNu"; }
	protected:
		/** SVM problem */
		svm_problem problem;
		/** SVM param */
		svm_parameter param;

		/** SVM model */
		struct svm_model* model;
};
#endif

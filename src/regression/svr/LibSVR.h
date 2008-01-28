/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVR_H___
#define _LIBSVR_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"

#include <stdio.h>

/** class LibSVR */
class CLibSVR : public CSVM
{
	public:
		/** default constructor */
		CLibSVR();

		/** constructor
		 *
		 * @param C constant C
		 * @param epsilon epsilon
		 * @param k kernel
		 * @param lab labels
		 */
		CLibSVR(DREAL C, DREAL epsilon, CKernel* k, CLabels* lab);
		virtual ~CLibSVR();

		/** train regression
		 *
		 * @return if training was successful
		 */
		virtual bool train();

		/** get classifier type
		 *
		 * @return classifie type LIBSVR
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIBSVR; }

	protected:
		/** SVM problem */
		svm_problem problem;
		/** SVM parameter */
		svm_parameter param;

		/** SVM model */
		struct svm_model* model;
};
#endif


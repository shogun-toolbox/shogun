/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVM_H___
#define _LIBSVM_H___

#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/external/shogun_libsvm.h>

namespace shogun
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum LIBSVM_SOLVER_TYPE
{
	LIBSVM_C_SVC = 1,
	LIBSVM_NU_SVC = 2
};
#endif
/** @brief LibSVM */
class CLibSVM : public CSVM
{
	public:
		/** Default constructor, create a C-SVC svm */
		CLibSVM();

		/** Constructor
		 *
		 * @param st solver type C or NU SVC
		 */
		CLibSVM(LIBSVM_SOLVER_TYPE st);

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 * @param st solver type to use, C-SVC or nu-SVC
		 */
		CLibSVM(float64_t C, CKernel* k, CLabels* lab,
				LIBSVM_SOLVER_TYPE st=LIBSVM_C_SVC);

		virtual ~CLibSVM();

		/** get classifier type
		 *
		 * @return classifier type LIBSVM
		 */
		virtual EMachineType get_classifier_type() { return CT_LIBSVM; }

		/** @return object name */
		virtual const char* get_name() const { return "LibSVM"; }

	protected:
		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	protected:
		/** SVM problem */
		svm_problem problem;
		/** SVM param */
		svm_parameter param;
		/** SVM model */
		struct svm_model* model;

		/** solver type */
		LIBSVM_SOLVER_TYPE solver_type;
};
}
#endif

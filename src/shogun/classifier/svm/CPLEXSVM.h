/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CPLEXSVM_H___
#define _CPLEXSVM_H___
#include <shogun/lib/common.h>

#ifdef USE_CPLEX
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/Cache.h>

namespace shogun
{
/** @brief CplexSVM a SVM solver implementation based on cplex (unfinished). */
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CCPLEXSVM : public CSVM
{
	public:
		CCPLEXSVM();
		virtual ~CCPLEXSVM();

		virtual inline EClassifierType get_classifier_type() { return CT_CPLEXSVM; }

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
};
}
#endif
#endif

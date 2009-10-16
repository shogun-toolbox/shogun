/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague 
 */

#ifndef _GMNPSVM_H___
#define _GMNPSVM_H___

#include <vector>

#include "lib/common.h"
#include "classifier/svm/MultiClassSVM.h"
#include "features/Features.h"

namespace shogun
{
/** @brief Class GMNPSVM implements a one vs. rest MultiClass SVM.
 *
 * It uses CGMNPLib for training (in true multiclass-SVM fashion).
 */
class CGMNPSVM : public CMultiClassSVM
{
	public:
		/** default constructor */
		CGMNPSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CGMNPSVM(float64_t C, CKernel* k, CLabels* lab);
		virtual ~CGMNPSVM();

		/** train SVM
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** get classifier type
		 *
		 * @return classifier type GMNPSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_GMNPSVM; }
		
		void getbasealphas(::std::vector< ::std::vector<float64_t> > & basealphas2);

		/** @return object name */
		inline virtual const char* get_name() const { return "GMNPSVM"; }
		
	protected: 
		::std::vector< ::std::vector<float64_t> > basealphas; // is the basic untransformed alpha, needed for MKL 
};
}
#endif

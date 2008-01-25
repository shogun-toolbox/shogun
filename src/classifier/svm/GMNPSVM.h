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

#include "lib/common.h"
#include "classifier/svm/MultiClassSVM.h"

/** class GMNPSVM */
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
		CGMNPSVM(DREAL C, CKernel* k, CLabels* lab);
		virtual ~CGMNPSVM();

		/** train SVM */
		virtual bool train();

		/** get classifier type
		 *
		 * @return classifier type GMNPSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_GMNPSVM; }
};
#endif

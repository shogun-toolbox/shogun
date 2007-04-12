/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2007 Center for Machine Perception, CTU FEL Prague 
 */

#ifndef _GMNPSVM_H___
#define _GMNPSVM_H___

#include "lib/common.h"
#include "classifier/svm/MultiClassSVM.h"

class CGMNPSVM : public CMultiClassSVM
{
	public:
		CGMNPSVM();
		CGMNPSVM(DREAL C, CKernel* k, CLabels* lab);
		virtual ~CGMNPSVM();
		virtual bool train();
		inline EClassifierType get_classifier_type() { return CT_GMNPSVM; }
};
#endif

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Vojtech Franc 
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SVMOCAS_H___
#define _SVMOCAS_H___

#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

enum E_SVM_TYPE
{
	E_SVMOCAS = 1,
	E_SVMPERF = 2
};

class CSVMOcas : public CSparseLinearClassifier
{
	public:
		CSVMOcas(E_SVM_TYPE);
		CSVMOcas(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);
		virtual ~CSVMOcas();

		virtual inline EClassifierType get_classifier_type() { return CT_SVMOCAS; }
		virtual bool train();

		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

		inline void set_epsilon(DREAL eps) { epsilon=eps; }
		inline DREAL get_epsilon() { return epsilon; }

	protected:
		DREAL C1;
		DREAL C2;
		DREAL epsilon;
		E_SVM_TYPE method;
};
#endif


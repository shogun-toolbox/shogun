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

#include "classifier/svm/SVMPerf.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

CSVMPerf::CSVMPerf() : CSparseLinearClassifier(), C1(1), C2(1), epsilon(1e-5)
{
}

CSVMPerf::CSVMPerf(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
	: CSparseLinearClassifier(), C1(C), C2(C), epsilon(1e-5)
{
	CSparseLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSVMPerf::~CSVMPerf()
{
}


bool CSVMPerf::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());

	INT num_train_labels=get_labels()->get_num_labels();
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);

	delete[] w;
	w=new DREAL[num_feat];
	ASSERT(w);
	bias=0;

	return true;
}

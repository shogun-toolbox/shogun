/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/SVMLin.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "classifier/svm/ssl.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

CSVMLin::CSVMLin() : CSparseLinearClassifier(), C1(1), C2(1)
{
}

CSVMLin::CSVMLin(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
	: CSparseLinearClassifier(), C1(C), C2(C)
{
	CSparseLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSVMLin::~CSVMLin()
{
}

bool CSVMLin::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());

	INT num_train_labels=0;
	DREAL* train_labels=get_labels()->get_labels(num_train_labels);
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	w=new DREAL[num_feat];
	ASSERT(w);

	struct options Options;
	struct data Data;
	struct vector_double Weights;
	struct vector_double Outputs;

	Data.m=num_vec;
	Data.l=num_vec;
	Data.u=0; 
	Data.n=num_feat;
	Data.nz=num_feat;
	Data.Y=train_labels;
	Data.features=get_features();
	Data.C = new double[Data.l];

	Options.algo = SVM;
	Options.algo = 1;
	Options.lambda=1.0;
	Options.lambda_u=1.0;
	Options.S=10000;
	Options.R=0.5;
	Options.epsilon = 1e-5; //FIXME
	Options.cgitermax=10000;
	Options.mfnitermax=50;
	Options.Cp = get_C1();
	Options.Cn = get_C2();


	for(int i=0;i<num_vec;i++)
	{
		if(train_labels[i]>0) 
			Data.C[i]=get_C1(); 
		else 
			Data.C[i]=get_C2();
	}
	ssl_train(&Data, &Options, &Weights, &Outputs);

	CMath::display_vector(Weights.vec, Weights.d, "weights");
	CMath::display_vector(Outputs.vec, Outputs.d, "outputs");
	return false;
}

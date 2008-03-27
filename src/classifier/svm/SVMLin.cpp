/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2008 Soeren Sonnenburg
 * Copyright (C) 2006-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/SVMLin.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "classifier/svm/ssl.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

CSVMLin::CSVMLin() : CSparseLinearClassifier(), C1(1), C2(1), epsilon(1e-5), use_bias(true)
{
}

CSVMLin::CSVMLin(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
	: CSparseLinearClassifier(), C1(C), C2(C), epsilon(1e-5), use_bias(true)
{
	set_features(traindat);
	set_labels(trainlab);
}


CSVMLin::~CSVMLin()
{
}

bool CSVMLin::train()
{
	ASSERT(labels);
	ASSERT(get_features());

	INT num_train_labels=0;
	DREAL* train_labels=labels->get_labels(num_train_labels);
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;

	struct options Options;
	struct data Data;
	struct vector_double Weights;
	struct vector_double Outputs;

	Data.l=num_vec;
	Data.m=num_vec;
	Data.u=0; 
	Data.n=num_feat+1;
	Data.nz=num_feat+1;
	Data.Y=train_labels;
	Data.features=get_features();
	Data.C = new double[Data.l];

	Options.algo = SVM;
	Options.lambda=1/(2*get_C1());
	Options.lambda_u=1/(2*get_C1());
	Options.S=10000;
	Options.R=0.5;
	Options.epsilon = get_epsilon();
	Options.cgitermax=10000;
	Options.mfnitermax=50;
	Options.Cp = get_C2()/get_C1();
	Options.Cn = 1;
	
	if (use_bias)
		Options.bias=1.0;
	else
		Options.bias=0.0;

	for(int i=0;i<num_vec;i++)
	{
		if(train_labels[i]>0) 
			Data.C[i]=Options.Cp;
		else 
			Data.C[i]=Options.Cn;
	}
	ssl_train(&Data, &Options, &Weights, &Outputs);
	ASSERT(Weights.vec && Weights.d == num_feat+1);

	DREAL sgn=train_labels[0];
	for (INT i=0; i<num_feat+1; i++)
		Weights.vec[i]*=sgn;

	CSparseLinearClassifier::set_w(Weights.vec, num_feat);
	CSparseLinearClassifier::set_bias(Weights.vec[num_feat]);

	delete[] Data.C;
	delete[] train_labels;
	delete[] Outputs.vec;
	return true;
}

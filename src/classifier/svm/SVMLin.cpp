/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/ssl.h"
#include "classifier/svm/SVMLin.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"

CSVMLin::CSVMLin() : CLinearClassifier()
{
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

	Data.val=0; //FIXME 
	Data.rowptr=0; //FIXME 
	Data.colind=0; //FIXME 
	
	Data.C = new double[Data.l];

	for(int i=0;i<num_vec;i++)
	{
		if(train_labels[i]>0) 
			Data.C[i]=get_C1(); 
		else 
			Data.C[i]=get_C2();
	}
	ssl_train(&Data, &Options, &Weights, &Outputs);

	return false;
}

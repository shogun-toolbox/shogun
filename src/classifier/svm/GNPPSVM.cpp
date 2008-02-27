/* 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague 
 */

#include "lib/io.h"
#include "classifier/svm/GNPPSVM.h"
#include "classifier/svm/gnpplib.h"

#define INDEX(ROW,COL,DIM) (((COL)*(DIM))+(ROW)) 

CGNPPSVM::CGNPPSVM() : CSVM()
{
}

CGNPPSVM::CGNPPSVM(DREAL C, CKernel* k, CLabels* lab) : CSVM(C, k, lab)
{
}

CGNPPSVM::~CGNPPSVM()
{
}

bool CGNPPSVM::train()
{

	ASSERT(get_labels() && get_labels()->get_num_labels());
	INT num_data = get_labels()->get_num_labels();
	SG_INFO( "%d trainlabels\n", num_data);

	DREAL* vector_y = new double[num_data];
	ASSERT(vector_y);


	for (int i=0; i<num_data; i++)
	{
		if (get_labels()->get_label(i) == +1)
			vector_y[i]= 1;
		else if (get_labels()->get_label(i) == -1)
			vector_y[i]= 2;
		else
			SG_ERROR("label unknown (%f)\n", get_labels()->get_label(i));
	}

	ASSERT(get_kernel());

	DREAL C = get_C1();
	INT tmax = 1000000000;
	DREAL tolabs = 0;
	DREAL tolrel = epsilon;

	DREAL reg_const=0;
	if( C!=0 )
		reg_const = 1/C; 

	DREAL* diagK = new DREAL[num_data];
	ASSERT(diagK);

	for(INT i = 0; i < num_data; i++ ) {
		diagK[i] = 2*get_kernel()->kernel(i,i) + reg_const;
	}

	DREAL* alpha = new DREAL[num_data];
	ASSERT(alpha);
	DREAL* vector_c = new DREAL[num_data];
	ASSERT(vector_c);

	memset(vector_c,0,num_data*sizeof(DREAL));

	DREAL thlb = 10000000000.0;
	INT t = 0;
	DREAL* History = NULL;
	INT verb = 0;
	DREAL aHa11, aHa22;

	CGNPPLib npp(vector_y,get_kernel(),num_data, reg_const);

	npp.gnpp_imdm(diagK, vector_c, vector_y, num_data, 
			tmax, tolabs, tolrel, thlb, alpha, &t, &aHa11, &aHa22, 
			&History, verb ); 

	INT num_sv = 0;
	DREAL nconst = History[INDEX(1,t,2)];
	DREAL trnerr = 0; /* counter of training error */

	for(INT i = 0; i < num_data; i++ )
	{
		if( alpha[i] != 0 ) num_sv++;
		if(vector_y[i] == 1) 
		{
			alpha[i] = alpha[i]*2/nconst;
			if( alpha[i]/(2*C) >= 1 ) trnerr++;
		}
		else
		{
			alpha[i] = -alpha[i]*2/nconst;
			if( alpha[i]/(2*C) <= -1 ) trnerr++;
		}
	}

	DREAL b = 0.5*(aHa22 - aHa11)/nconst;;

	create_new_model(num_sv);
	CSVM::set_objective(nconst);

	set_bias(b);
	INT j = 0;
	for (int i=0; i<num_data; i++)
	{
		if( alpha[i] !=0)
		{
			set_support_vector(j, i);
			set_alpha(j, alpha[i]);
			j++;
		}
	}

	delete[] vector_c;
	delete[] alpha;
	delete[] diagK;
	delete[] vector_y;
	delete[] History;

	return true;
}

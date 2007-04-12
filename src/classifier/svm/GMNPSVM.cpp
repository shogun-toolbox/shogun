/* 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2007 Center for Machine Perception, CTU FEL Prague 
 */

#include "lib/io.h"
#include "classifier/svm/GMNPSVM.h"
#include "classifier/svm/gmnplib.h"

#define INDEX(ROW,COL,DIM) (((COL)*(DIM))+(ROW)) 
#define MINUS_INF INT_MIN
#define PLUS_INF  INT_MAX
#define KDELTA(A,B) (A==B)
#define KDELTA4(A1,A2,A3,A4) ((A1==A2)||(A1==A3)||(A1==A4)||(A2==A3)||(A2==A4)||(A3==A4))

CGMNPSVM::CGMNPSVM() : CMultiClassSVM()
{
}

CGMNPSVM::CGMNPSVM(DREAL C, CKernel* k, CLabels* lab) : CMultiClassSVM(C, k, lab)
{
}

CGMNPSVM::~CGMNPSVM()
{
}

bool CGMNPSVM::train()
{
	ASSERT(get_labels() && get_labels()->get_num_labels());
	INT num_data = get_labels()->get_num_labels();
	INT num_classes = get_labels()->get_num_classes();
	INT num_virtual_data= num_data*(num_classes-1);

	SG_INFO( "%d trainlabels, %d classes\n", num_data, num_classes);

	DREAL* vector_y = new double[num_data];
	ASSERT(vector_y);

	for (int i=0; i<num_data; i++)
		vector_y[i]= get_labels()->get_label(i)+1;

	ASSERT(get_kernel());

	DREAL C = get_C1();
	INT tmax = 1000000000;
	DREAL tolabs = 0;
	DREAL tolrel = epsilon;

	DREAL reg_const=0;
	if( C!=0 )
		reg_const = 1/(2*C); 


	DREAL* alpha = new DREAL[num_virtual_data];
	ASSERT(alpha);
	DREAL* vector_c = new DREAL[num_virtual_data];
	ASSERT(vector_c);

	memset(vector_c,0,num_virtual_data*sizeof(DREAL));

	DREAL thlb = 10000000000.0;
	INT t = 0;
	DREAL* History = NULL;
	INT verb = 0;

	CGMNPLib mnp(vector_y,get_kernel(),num_data, num_virtual_data, num_classes, reg_const);

	mnp.gmnp_imdm(vector_c, num_virtual_data, tmax,
			tolabs, tolrel, thlb, alpha, &t, &History, verb );

	/* matrix alpha [num_classes x num_data] */
	DREAL* all_alphas= new DREAL[num_classes*num_data];
	memset(all_alphas,0,num_classes*num_data*sizeof(DREAL));

	/* bias vector b [num_classes x 1] */
	DREAL* all_bs=new DREAL[num_classes];
	memset(all_bs,0,num_classes*sizeof(DREAL));

	/* compute alpha/b from virt_data */
	for(INT i=0; i < num_classes; i++ )
	{
		for(INT j=0; j < num_virtual_data; j++ )
		{
			INT inx1=0;
			INT inx2=0;

			mnp.get_indices2( &inx1, &inx2, j );

			all_alphas[(inx1*num_classes)+i] += 
				alpha[j]*(KDELTA(vector_y[inx1],i+1)+KDELTA(i+1,inx2));
			all_bs[i] += alpha[j]*(KDELTA(vector_y[inx1],i+1)-KDELTA(i+1,inx2));
		}
	}

	create_multiclass_svm(num_classes);

	for (INT i=0; i<num_classes; i++)
	{
		INT num_sv=0;
		for (INT j=0; j<num_data; j++)
		{
			if (all_alphas[j*num_classes+i] != 0)
				num_sv++;
		}
		ASSERT(num_sv>0);
		SG_DEBUG("svm[%d] has %d sv, b=%f\n", i, num_sv, all_bs[i]);

		CSVM* svm=new CSVM(num_sv);

		INT k=0;
		for (INT j=0; j<num_data; j++)
		{
			if (all_alphas[j*num_classes+i] != 0)
			{
				if (i==vector_y[j]-1)
					svm->set_alpha(k, all_alphas[j*num_classes+i]);
				else
					svm->set_alpha(k, -all_alphas[j*num_classes+i]);

				svm->set_support_vector(k, j);
				k++;
			}
		}

		svm->set_bias(all_bs[i]);
		set_svm(i, svm);
	}

	delete[] vector_c;
	delete[] alpha;
	delete[] all_alphas;
	delete[] all_bs;
	delete[] vector_y;
	delete[] History;

	return true;
}

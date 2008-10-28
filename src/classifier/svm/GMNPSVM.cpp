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
#include "classifier/svm/GMNPSVM.h"
#include "classifier/svm/gmnplib.h"

#define INDEX(ROW,COL,DIM) (((COL)*(DIM))+(ROW)) 
#define MINUS_INF INT_MIN
#define PLUS_INF  INT_MAX
#define KDELTA(A,B) (A==B)
#define KDELTA4(A1,A2,A3,A4) ((A1==A2)||(A1==A3)||(A1==A4)||(A2==A3)||(A2==A4)||(A3==A4))

CGMNPSVM::CGMNPSVM()
: CMultiClassSVM(ONE_VS_REST)
{
}

CGMNPSVM::CGMNPSVM(float64_t C, CKernel* k, CLabels* lab)
: CMultiClassSVM(ONE_VS_REST, C, k, lab)
{
}

CGMNPSVM::~CGMNPSVM()
{
}

bool CGMNPSVM::train()
{
	ASSERT(kernel);
	ASSERT(labels && labels->get_num_labels());

	int32_t num_data = labels->get_num_labels();
	int32_t num_classes = labels->get_num_classes();
	int32_t num_virtual_data= num_data*(num_classes-1);

	SG_INFO( "%d trainlabels, %d classes\n", num_data, num_classes);

	float64_t* vector_y = new double[num_data];
	for (int32_t i=0; i<num_data; i++)
		vector_y[i]= labels->get_label(i)+1;

	float64_t C = get_C1();
	int32_t tmax = 1000000000;
	float64_t tolabs = 0;
	float64_t tolrel = epsilon;

	float64_t reg_const=0;
	if( C!=0 )
		reg_const = 1/(2*C);


	float64_t* alpha = new float64_t[num_virtual_data];
	float64_t* vector_c = new float64_t[num_virtual_data];
	memset(vector_c, 0, num_virtual_data*sizeof(float64_t));

	float64_t thlb = 10000000000.0;
	int32_t t = 0;
	float64_t* History = NULL;
	int32_t verb = 0;

	CGMNPLib mnp(vector_y,kernel,num_data, num_virtual_data, num_classes, reg_const);

	mnp.gmnp_imdm(vector_c, num_virtual_data, tmax,
			tolabs, tolrel, thlb, alpha, &t, &History, verb );

	/* matrix alpha [num_classes x num_data] */
	float64_t* all_alphas= new float64_t[num_classes*num_data];
	memset(all_alphas,0,num_classes*num_data*sizeof(float64_t));

	/* bias vector b [num_classes x 1] */
	float64_t* all_bs=new float64_t[num_classes];
	memset(all_bs,0,num_classes*sizeof(float64_t));

	/* compute alpha/b from virt_data */
	for(int32_t i=0; i < num_classes; i++ )
	{
		for(int32_t j=0; j < num_virtual_data; j++ )
		{
			int32_t inx1=0;
			int32_t inx2=0;

			mnp.get_indices2( &inx1, &inx2, j );

			all_alphas[(inx1*num_classes)+i] += 
				alpha[j]*(KDELTA(vector_y[inx1],i+1)-KDELTA(i+1,inx2));
			all_bs[i] += alpha[j]*(KDELTA(vector_y[inx1],i+1)-KDELTA(i+1,inx2));
		}
	}

	create_multiclass_svm(num_classes);

	for (int32_t i=0; i<num_classes; i++)
	{
		int32_t num_sv=0;
		for (int32_t j=0; j<num_data; j++)
		{
			if (all_alphas[j*num_classes+i] != 0)
				num_sv++;
		}
		ASSERT(num_sv>0);
		SG_DEBUG("svm[%d] has %d sv, b=%f\n", i, num_sv, all_bs[i]);

		CSVM* svm=new CSVM(num_sv);

		int32_t k=0;
		for (int32_t j=0; j<num_data; j++)
		{
			if (all_alphas[j*num_classes+i] != 0)
			{
				svm->set_alpha(k, all_alphas[j*num_classes+i]);
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

/* 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague 
 */

#include <io/SGIO.h>
#include <classifier/svm/GNPPSVM.h>
#include <classifier/svm/GNPPLib.h>
#include <labels/BinaryLabels.h>

using namespace shogun;
#define INDEX(ROW,COL,DIM) (((COL)*(DIM))+(ROW)) 

CGNPPSVM::CGNPPSVM()
: CSVM()
{
}

CGNPPSVM::CGNPPSVM(float64_t C, CKernel* k, CLabels* lab)
: CSVM(C, k, lab)
{
}

CGNPPSVM::~CGNPPSVM()
{
}

bool CGNPPSVM::train_machine(CFeatures* data)
{
	ASSERT(kernel)
	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(m_labels->get_label_type() == LT_BINARY)

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")
		kernel->init(data, data);
	}

	int32_t num_data=m_labels->get_num_labels();
	SG_INFO("%d trainlabels\n", num_data)

	float64_t* vector_y = SG_MALLOC(float64_t, num_data);
	for (int32_t i=0; i<num_data; i++)
	{
		float64_t lab=((CBinaryLabels*) m_labels)->get_label(i);
		if (lab==+1)
			vector_y[i]=1;
		else if (lab==-1)
			vector_y[i]=2;
		else
			SG_ERROR("label unknown (%f)\n", lab)
	}

	float64_t C=get_C1();
	int32_t tmax=1000000000;
	float64_t tolabs=0;
	float64_t tolrel=epsilon;

	float64_t reg_const=0;
	if (C!=0)
		reg_const=1/C;

	float64_t* diagK=SG_MALLOC(float64_t, num_data);
	for(int32_t i=0; i<num_data; i++) {
		diagK[i]=2*kernel->kernel(i,i)+reg_const;
	}

	float64_t* alpha=SG_MALLOC(float64_t, num_data);
	float64_t* vector_c=SG_MALLOC(float64_t, num_data);
	memset(vector_c, 0, num_data*sizeof(float64_t));

	float64_t thlb=10000000000.0;
	int32_t t=0;
	float64_t* History=NULL;
	int32_t verb=0;
	float64_t aHa11, aHa22;

	CGNPPLib npp(vector_y,kernel,num_data, reg_const);

	npp.gnpp_imdm(diagK, vector_c, vector_y, num_data, 
			tmax, tolabs, tolrel, thlb, alpha, &t, &aHa11, &aHa22, 
			&History, verb ); 

	int32_t num_sv = 0;
	float64_t nconst = History[INDEX(1,t,2)];
	float64_t trnerr = 0; /* counter of training error */

	for(int32_t i = 0; i < num_data; i++ )
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

	float64_t b = 0.5*(aHa22 - aHa11)/nconst;;

	create_new_model(num_sv);
	CSVM::set_objective(nconst);

	set_bias(b);
	int32_t j = 0;
	for (int32_t i=0; i<num_data; i++)
	{
		if( alpha[i] !=0)
		{
			set_support_vector(j, i);
			set_alpha(j, alpha[i]);
			j++;
		}
	}

	SG_FREE(vector_c);
	SG_FREE(alpha);
	SG_FREE(diagK);
	SG_FREE(vector_y);
	SG_FREE(History);

	return true;
}

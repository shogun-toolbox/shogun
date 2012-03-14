/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague
 */

#include <shogun/io/SGIO.h>
#include <shogun/classifier/svm/GMNPSVM.h>
#include <shogun/classifier/svm/GMNPLib.h>

#define INDEX(ROW,COL,DIM) (((COL)*(DIM))+(ROW))
#define MINUS_INF INT_MIN
#define PLUS_INF  INT_MAX
#define KDELTA(A,B) (A==B)
#define KDELTA4(A1,A2,A3,A4) ((A1==A2)||(A1==A3)||(A1==A4)||(A2==A3)||(A2==A4)||(A3==A4))

using namespace shogun;

CGMNPSVM::CGMNPSVM()
: CMultiClassSVM(ONE_VS_REST)
{
	init();
}

CGMNPSVM::CGMNPSVM(float64_t C, CKernel* k, CLabels* lab)
: CMultiClassSVM(ONE_VS_REST, C, k, lab)
{
	init();
}

CGMNPSVM::~CGMNPSVM()
{
	if (m_basealphas != NULL) SG_FREE(m_basealphas);
}

void
CGMNPSVM::init()
{
	m_parameters->add_matrix(&m_basealphas,
							 &m_basealphas_y, &m_basealphas_x,
							 "m_basealphas",
							 "Is the basic untransformed alpha.");

	m_basealphas = NULL, m_basealphas_y = 0, m_basealphas_x = 0;
}

bool CGMNPSVM::train_machine(CFeatures* data)
{
	ASSERT(kernel);
	ASSERT(m_labels && m_labels->get_num_labels());

	if (data)
	{
		if (data->get_num_vectors() != m_labels->get_num_labels())
		{
			SG_ERROR("Numbert of vectors (%d) does not match number of labels (%d)\n",
					data->get_num_vectors(), m_labels->get_num_labels());
		}
		kernel->init(data, data);
	}

	int32_t num_data = m_labels->get_num_labels();
	int32_t num_classes = m_labels->get_num_classes();
	int32_t num_virtual_data= num_data*(num_classes-1);

	SG_INFO( "%d trainlabels, %d classes\n", num_data, num_classes);

	float64_t* vector_y = SG_MALLOC(float64_t, num_data);
	for (int32_t i=0; i<num_data; i++)
	{
		vector_y[i] = m_labels->get_label(i)+1;

	}

	float64_t C = get_C1();
	int32_t tmax = 1000000000;
	float64_t tolabs = 0;
	float64_t tolrel = epsilon;

	float64_t reg_const=0;
	if( C!=0 )
		reg_const = 1/(2*C);


	float64_t* alpha = SG_MALLOC(float64_t, num_virtual_data);
	float64_t* vector_c = SG_MALLOC(float64_t, num_virtual_data);
	memset(vector_c, 0, num_virtual_data*sizeof(float64_t));

	float64_t thlb = 10000000000.0;
	int32_t t = 0;
	float64_t* History = NULL;
	int32_t verb = 0;

	CGMNPLib mnp(vector_y,kernel,num_data, num_virtual_data, num_classes, reg_const);

	mnp.gmnp_imdm(vector_c, num_virtual_data, tmax,
				  tolabs, tolrel, thlb, alpha, &t, &History, verb);

	/* matrix alpha [num_classes x num_data] */
	float64_t* all_alphas= SG_MALLOC(float64_t, num_classes*num_data);
	memset(all_alphas,0,num_classes*num_data*sizeof(float64_t));

	/* bias vector b [num_classes x 1] */
	float64_t* all_bs=SG_MALLOC(float64_t, num_classes);
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

	if (m_basealphas != NULL) SG_FREE(m_basealphas);
	m_basealphas_y = num_classes, m_basealphas_x = num_data;
	m_basealphas = SG_MALLOC(float64_t, m_basealphas_y*m_basealphas_x);
	for (index_t i=0; i<m_basealphas_y*m_basealphas_x; i++)
		m_basealphas[i] = 0.0;

	for(index_t j=0; j<num_virtual_data; j++)
	{
		index_t inx1=0, inx2=0;

		mnp.get_indices2(&inx1, &inx2, j);
		m_basealphas[inx1*m_basealphas_y + (inx2-1)] = alpha[j];
	}

	SG_FREE(vector_c);
	SG_FREE(alpha);
	SG_FREE(all_alphas);
	SG_FREE(all_bs);
	SG_FREE(vector_y);
	SG_FREE(History);

	return true;
}

float64_t*
CGMNPSVM::get_basealphas_ptr(index_t* y, index_t* x)
{
	if (y == NULL || x == NULL) return NULL;

	*y = m_basealphas_y, *x = m_basealphas_x;
	return m_basealphas;
}

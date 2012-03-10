/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009-2012 Vojtech Franc and Soeren Sonnenburg
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2009-2012 Vojtech Franc and Soeren Sonnenburg
 */

#include <shogun/classifier/svm/MulticlassOCAS.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

struct mocas_data
{
	CDotFeatures* features;
	float64_t* W;
	float64_t* oldW;
	float64_t* full_A;
	float64_t* data_y;
	uint32_t nY;
	uint32_t nData;
	uint32_t nDim;
	float64_t* new_a;
};

CMulticlassOCAS::CMulticlassOCAS() : 
	CLinearMulticlassMachine()
{
	register_parameters();
	set_C(1.0);
	set_epsilon(1e-2);
	set_max_iter(1000000);
	set_method(1);
	set_buf_size(5000);
}

CMulticlassOCAS::CMulticlassOCAS(float64_t C, CDotFeatures* train_features, CLabels* train_labels) :
	CLinearMulticlassMachine(ONE_VS_REST_STRATEGY, train_features, NULL, train_labels), m_C(C)
{
	register_parameters();
	set_epsilon(1e-2);
	set_max_iter(1000000);
	set_method(1);
	set_buf_size(5000);
}

void CMulticlassOCAS::register_parameters()
{
	m_parameters->add(&m_C, "m_C", "regularization constant");
	m_parameters->add(&m_epsilon, "m_epsilon", "solver relative tolerance");
	m_parameters->add(&m_max_iter, "m_max_iter", "max number of iterations");
	m_parameters->add(&m_method, "m_method", "used solver method");
	m_parameters->add(&m_buf_size, "m_buf_size", "buffer size");
}

CMulticlassOCAS::~CMulticlassOCAS()
{
}

bool CMulticlassOCAS::train_machine(CFeatures* data)
{
	if (data)
		set_features((CDotFeatures*)data);

	int32_t num_vectors = m_features->get_num_vectors();
	int32_t num_classes = labels->get_num_classes();
	int32_t num_features = m_features->get_dim_feature_space();

	float64_t C = m_C;
	float64_t* data_y = labels->get_labels().vector;
	uint32_t nY = num_classes;
	uint32_t nData = num_vectors;
	float64_t TolRel = m_epsilon;
	float64_t TolAbs = 0.0;
	float64_t QPBound = 0.0;
	float64_t MaxTime = max_train_time;
	uint32_t BufSize = m_buf_size;
	uint8_t Method = m_method;

	mocas_data user_data;
	user_data.features = m_features;
	user_data.W = SG_MALLOC(float64_t, num_features*num_classes);
	user_data.oldW = SG_MALLOC(float64_t, num_features*num_classes);
	user_data.new_a = SG_MALLOC(float64_t, num_features*num_classes);
	user_data.full_A = SG_MALLOC(float64_t, num_features*num_classes*m_buf_size);
	user_data.data_y = data_y;
	user_data.nY = num_classes;
	user_data.nDim = num_features;
	user_data.nData = num_vectors;

	ocas_return_value_T ocas = 
		msvm_ocas_solver(C, data_y, nY, nData, TolRel, TolAbs, 
		                 QPBound, MaxTime, BufSize, Method,
		                 &CMulticlassOCAS::msvm_full_compute_W,
		                 &CMulticlassOCAS::msvm_update_W,
		                 &CMulticlassOCAS::msvm_full_add_new_cut,
		                 &CMulticlassOCAS::msvm_full_compute_output,
		                 &CMulticlassOCAS::msvm_sort_data,
		                 &CMulticlassOCAS::msvm_print,
		                 &user_data);

	clear_machines();
	m_machines = SGVector<CMachine*>(num_classes);
	for (int32_t i=0; i<num_classes; i++)
	{
		CLinearMachine* machine = new CLinearMachine();
		machine->set_w(SGVector<float64_t>(&user_data.W[i*num_features],num_features).clone());

		m_machines[i] = machine;
	}

	SG_FREE(user_data.W);
	SG_FREE(user_data.oldW);
	SG_FREE(user_data.new_a);
	SG_FREE(user_data.full_A);

	return true;
}

float64_t CMulticlassOCAS::msvm_update_W(float64_t t, void* user_data)
{
	float64_t* W = ((mocas_data*)user_data)->W;
	float64_t* oldW = ((mocas_data*)user_data)->oldW;
	uint32_t nY = ((mocas_data*)user_data)->nY;
	uint32_t nDim = ((mocas_data*)user_data)->nDim;

	for(uint32_t j=0; j < nY*nDim; j++) 
		W[j] = oldW[j]*(1-t) + t*W[j];

	float64_t sq_norm_W = CMath::dot(W,W,nDim*nY);

	return sq_norm_W;
}

void CMulticlassOCAS::msvm_full_compute_W(float64_t *sq_norm_W, float64_t *dp_WoldW, 
                                          float64_t *alpha, uint32_t nSel, void* user_data)
{
	float64_t* W = ((mocas_data*)user_data)->W;
	float64_t* oldW = ((mocas_data*)user_data)->oldW;
	float64_t* full_A = ((mocas_data*)user_data)->full_A;
	uint32_t nY = ((mocas_data*)user_data)->nY;
	uint32_t nDim = ((mocas_data*)user_data)->nDim;

	uint32_t i,j;

	memcpy(oldW, W, sizeof(float64_t)*nDim*nY); 
	memset(W, 0, sizeof(float64_t)*nDim*nY);

	for(i=0; i<nSel; i++)
	{
		if(alpha[i] > 0)
		{
			for(j=0; j<nDim*nY; j++) 
				W[j] += alpha[i]*full_A[LIBOCAS_INDEX(j,i,nDim*nY)];
		}
	}

	*sq_norm_W = CMath::dot(W,W,nDim*nY);
	*dp_WoldW = CMath::dot(W,oldW,nDim*nY);

	return;
}

int CMulticlassOCAS::msvm_full_add_new_cut(float64_t *new_col_H, uint32_t *new_cut, 
                                           uint32_t nSel, void* user_data)
{
	float64_t* full_A = ((mocas_data*)user_data)->full_A;
	float64_t* new_a = ((mocas_data*)user_data)->new_a;
	float64_t* data_y = ((mocas_data*)user_data)->data_y;
	uint32_t nY = ((mocas_data*)user_data)->nY;
	uint32_t nDim = ((mocas_data*)user_data)->nDim;
	uint32_t nData = ((mocas_data*)user_data)->nData;
	CDotFeatures* features = ((mocas_data*)user_data)->features;

	float64_t sq_norm_a;
	uint32_t i, j, y, y2;

	memset(new_a, 0, sizeof(float64_t)*nDim*nY);

	for(i=0; i < nData; i++)
	{
		y = (uint32_t)(data_y[i]);
		y2 = (uint32_t)new_cut[i];
		if(y2 != y)
		{
			features->add_to_dense_vec(1.0,i,&new_a[nDim*y],nDim);
			features->add_to_dense_vec(-1.0,i,&new_a[nDim*y2],nDim);
		}
	}

	// compute new_a'*new_a and insert new_a to the last column of full_A
	sq_norm_a = CMath::dot(new_a,new_a,nDim*nY);
	for(j=0; j < nDim*nY; j++ ) 
		full_A[LIBOCAS_INDEX(j,nSel,nDim*nY)] = new_a[j];

	new_col_H[nSel] = sq_norm_a;
	for(i=0; i < nSel; i++) 
	{
		float64_t tmp = 0;

		for(j=0; j < nDim*nY; j++ ) 
			tmp += new_a[j]*full_A[LIBOCAS_INDEX(j,i,nDim*nY)];

		new_col_H[i] = tmp;
	}
	
	return 0;
}

int CMulticlassOCAS::msvm_full_compute_output(float64_t *output, void* user_data)
{
	float64_t* W = ((mocas_data*)user_data)->W;
	uint32_t nY = ((mocas_data*)user_data)->nY;
	uint32_t nDim = ((mocas_data*)user_data)->nDim;
	uint32_t nData = ((mocas_data*)user_data)->nData;
	CDotFeatures* features = ((mocas_data*)user_data)->features;

	uint32_t i, y;

	for(i=0; i < nData; i++) 
	{ 
		for(y=0; y < nY; y++)
			output[LIBOCAS_INDEX(y,i,nY)] = 
				features->dense_dot(i,&W[nDim*y],nDim);
	}

	return 0;
}

int CMulticlassOCAS::msvm_sort_data(float64_t* vals, float64_t* data, uint32_t size)
{
	CMath::qsort_index(vals, data, size);
	return 0;
}

void CMulticlassOCAS::msvm_print(ocas_return_value_T value)
{
	return;
}



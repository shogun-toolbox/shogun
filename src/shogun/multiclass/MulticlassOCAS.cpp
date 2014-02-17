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

#include <shogun/multiclass/MulticlassOCAS.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

struct mocas_data
{
	CDotFeatures* features;
	float64_t* W;
	float64_t* oldW;
	float64_t* full_A;
	float64_t* data_y;
	float64_t* output_values;
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
	CLinearMulticlassMachine(new CMulticlassOneVsRestStrategy(), train_features, NULL, train_labels), m_C(C)
{
	register_parameters();
	set_epsilon(1e-2);
	set_max_iter(1000000);
	set_method(1);
	set_buf_size(5000);
}

void CMulticlassOCAS::register_parameters()
{
	SG_ADD(&m_C, "m_C", "regularization constant", MS_AVAILABLE);
	SG_ADD(&m_epsilon, "m_epsilon", "solver relative tolerance", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iter, "m_max_iter", "max number of iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_method, "m_method", "used solver method", MS_NOT_AVAILABLE);
	SG_ADD(&m_buf_size, "m_buf_size", "buffer size", MS_NOT_AVAILABLE);
}

CMulticlassOCAS::~CMulticlassOCAS()
{
}

bool CMulticlassOCAS::train_machine(CFeatures* data)
{
	if (data)
		set_features((CDotFeatures*)data);

	ASSERT(m_features)
	ASSERT(m_labels)
	ASSERT(m_multiclass_strategy)

	int32_t num_vectors = m_features->get_num_vectors();
	int32_t num_classes = m_multiclass_strategy->get_num_classes();
	int32_t num_features = m_features->get_dim_feature_space();

	float64_t C = m_C;
	SGVector<float64_t> labels = ((CMulticlassLabels*) m_labels)->get_labels();
	uint32_t nY = num_classes;
	uint32_t nData = num_vectors;
	float64_t TolRel = m_epsilon;
	float64_t TolAbs = 0.0;
	float64_t QPBound = 0.0;
	float64_t MaxTime = m_max_train_time;
	uint32_t BufSize = m_buf_size;
	uint8_t Method = m_method;

	mocas_data user_data;
	user_data.features = m_features;
	user_data.W = SG_CALLOC(float64_t, (int64_t)num_features*num_classes);
	user_data.oldW = SG_CALLOC(float64_t, (int64_t)num_features*num_classes);
	user_data.new_a = SG_CALLOC(float64_t, (int64_t)num_features*num_classes);
	user_data.full_A = SG_CALLOC(float64_t, (int64_t)num_features*num_classes*m_buf_size);
	user_data.output_values = SG_CALLOC(float64_t, num_vectors);
	user_data.data_y = labels.vector;
	user_data.nY = num_classes;
	user_data.nDim = num_features;
	user_data.nData = num_vectors;

	ocas_return_value_T value =
	msvm_ocas_solver(C, labels.vector, nY, nData, TolRel, TolAbs,
	                 QPBound, MaxTime, BufSize, Method,
	                 &CMulticlassOCAS::msvm_full_compute_W,
	                 &CMulticlassOCAS::msvm_update_W,
	                 &CMulticlassOCAS::msvm_full_add_new_cut,
	                 &CMulticlassOCAS::msvm_full_compute_output,
	                 &CMulticlassOCAS::msvm_sort_data,
	                 &CMulticlassOCAS::msvm_print,
	                 &user_data);

	SG_DEBUG("Number of iterations [nIter] = %d \n",value.nIter)
	SG_DEBUG("Number of cutting planes [nCutPlanes] = %d \n",value.nCutPlanes)
	SG_DEBUG("Number of non-zero alphas [nNZAlpha] = %d \n",value.nNZAlpha)
	SG_DEBUG("Number of training errors [trn_err] = %d \n",value.trn_err)
	SG_DEBUG("Primal objective value [Q_P] = %f \n",value.Q_P)
	SG_DEBUG("Dual objective value [Q_D] = %f \n",value.Q_D)
	SG_DEBUG("Output time [output_time] = %f \n",value.output_time)
	SG_DEBUG("Sort time [sort_time] = %f \n",value.sort_time)
	SG_DEBUG("Add time [add_time] = %f \n",value.add_time)
	SG_DEBUG("W time [w_time] = %f \n",value.w_time)
	SG_DEBUG("QP solver time [qp_solver_time] = %f \n",value.qp_solver_time)
	SG_DEBUG("OCAS time [ocas_time] = %f \n",value.ocas_time)
	SG_DEBUG("Print time [print_time] = %f \n",value.print_time)
	SG_DEBUG("QP exit flag [qp_exitflag] = %d \n",value.qp_exitflag)
	SG_DEBUG("Exit flag [exitflag] = %d \n",value.exitflag)

	m_machines->reset_array();
	for (int32_t i=0; i<num_classes; i++)
	{
		CLinearMachine* machine = new CLinearMachine();
		machine->set_w(SGVector<float64_t>(&user_data.W[i*num_features],num_features,false).clone());

		m_machines->push_back(machine);
	}

	SG_FREE(user_data.W);
	SG_FREE(user_data.oldW);
	SG_FREE(user_data.new_a);
	SG_FREE(user_data.full_A);
	SG_FREE(user_data.output_values);

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

	float64_t sq_norm_W = SGVector<float64_t>::dot(W,W,nDim*nY);

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

	*sq_norm_W = SGVector<float64_t>::dot(W,W,nDim*nY);
	*dp_WoldW = SGVector<float64_t>::dot(W,oldW,nDim*nY);

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
	sq_norm_a = SGVector<float64_t>::dot(new_a,new_a,nDim*nY);
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
	float64_t* output_values = ((mocas_data*)user_data)->output_values;
	CDotFeatures* features = ((mocas_data*)user_data)->features;

	uint32_t i, y;

	for(y=0; y<nY; y++)
	{
		features->dense_dot_range(output_values,0,nData,NULL,&W[nDim*y],nDim,0.0);
		for (i=0; i<nData; i++)
			output[LIBOCAS_INDEX(y,i,nY)] = output_values[i];
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
}

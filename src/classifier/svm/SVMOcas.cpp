/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Vojtech Franc 
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/Time.h"
#include "base/Parallel.h"
#include "classifier/SparseLinearClassifier.h"
#include "classifier/svm/SVMOcas.h"
#include "classifier/svm/libocas.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

CSVMOcas::CSVMOcas(E_SVM_TYPE type) : CSparseLinearClassifier(), use_bias(false), bufsize(3000), C1(1), C2(1),
	epsilon(1e-3), method(type)
{
	w=NULL;
	old_w=NULL;
}

CSVMOcas::CSVMOcas(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
: CSparseLinearClassifier(), use_bias(false), bufsize(3000), C1(C), C2(C), epsilon(1e-3)
{
	w=NULL;
	old_w=NULL;
	method=SVM_OCAS;
	CSparseLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSVMOcas::~CSVMOcas()
{
}

bool CSVMOcas::train()
{
	SG_INFO("C=%f, epsilon=%f, bufsize=%d\n", get_C1(), get_epsilon(), bufsize);

	ASSERT(get_labels());
	ASSERT(get_features());

	INT num_train_labels=0;
	lab=get_labels()->get_labels(num_train_labels);
	w_dim=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	ASSERT(num_vec>0);

	delete[] w;
	w=new DREAL[w_dim];
	ASSERT(w);
	memset(w, 0, w_dim*sizeof(DREAL));

	delete[] old_w;
	old_w=new DREAL[w_dim];
	ASSERT(old_w);
	memset(old_w, 0, w_dim*sizeof(DREAL));
	bias=0;

	tmp_a_buf = new DREAL[w_dim];
	ASSERT(tmp_a_buf);
	memset(tmp_a_buf, 0, w_dim*sizeof(DREAL));

	cp_value = new DREAL*[bufsize];
	ASSERT(cp_value);
	memset(cp_value, 0, bufsize*sizeof(DREAL*));

	cp_index = new uint32_t*[bufsize];
	ASSERT(cp_index);
	memset(cp_index, 0, bufsize*sizeof(uint32_t*));

	cp_nz_dims = new uint32_t[bufsize];
	ASSERT(cp_nz_dims);
	memset(cp_nz_dims, 0, bufsize*sizeof(uint32_t*));

	double TolAbs=0;
	double QPBound=0;
	int Method=0;
	if (method == SVM_OCAS)
		Method = 1;
	ocas_return_value_T result = svm_ocas_solver( get_C1(), num_vec, get_epsilon(),
			TolAbs, QPBound, bufsize, Method, 
			&CSVMOcas::compute_W,
			&CSVMOcas::update_W, 
			&CSVMOcas::add_new_cut, 
			&CSVMOcas::compute_output,
			&CSVMOcas::sort,
			&printf,
			this);

	delete[] tmp_a_buf;

	uint32_t num_cut_planes = result.nCutPlanes;

	for (uint32_t i=0; i<num_cut_planes; i++)
	{
		delete[] cp_value[i];
		delete[] cp_index[i];
	}

	delete[] cp_value;
	cp_value=NULL;
	delete[] cp_index;
	cp_index=NULL;
	delete[] cp_nz_dims;
	cp_nz_dims=NULL;

	delete[] lab;
	lab=NULL;

	return true;
}

/*----------------------------------------------------------------------------------
  sq_norm_W = sparse_update_W( t ) does the following:

  W = oldW*(1-t) + t*W;
  sq_norm_W = W'*W;

  ---------------------------------------------------------------------------------*/
double CSVMOcas::update_W( double t, void* ptr )
{
  double sq_norm_W = 0;         
  CSVMOcas* o = (CSVMOcas*) ptr;
  uint32_t nDim = (uint32_t) o->w_dim;
  double* W=o->w;
  double* oldW=o->old_w;

  for(uint32_t j=0; j <nDim; j++)
  {
	  W[j] = oldW[j]*(1-t) + t*W[j];
	  sq_norm_W += W[j]*W[j];
  }          

  return( sq_norm_W );
}

/*----------------------------------------------------------------------------------
  sparse_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [sparse_A(:,1:nSel)'*new_a ; new_a'*new_a];
    sparse_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
void CSVMOcas::add_new_cut( double *new_col_H, 
                  uint32_t *new_cut, 
                  uint32_t cut_length, 
                  uint32_t nSel,
				  void* ptr)
{
	CSVMOcas* o = (CSVMOcas*) ptr;
	CSparseFeatures<DREAL>* f = o->get_features();
	uint32_t nDim=(uint32_t) o->w_dim;
	DREAL* y = o->lab;

	DREAL** c_val = o->cp_value;
	uint32_t** c_idx = o->cp_index;
	uint32_t* c_nzd = o->cp_nz_dims;

	double sq_norm_a;
	uint32_t i, j, nz_dims;

	/* temporary vector */
	double* new_a = o->tmp_a_buf;
	memset(new_a, 0, sizeof(double)*nDim);

	for(i=0; i < cut_length; i++) 
		f->add_to_dense_vec(y[new_cut[i]], new_cut[i], new_a, nDim);

	/* compute new_a'*new_a and count number of non-zerou dimensions */
	nz_dims = 0; 
	sq_norm_a = 0;
	for(j=0; j < nDim; j++ ) {
		if(new_a[j] != 0) {
			nz_dims++;
			sq_norm_a += new_a[j]*new_a[j];
		}
	}

	/* sparsify new_a and insert it to the last column of sparse_A */
	c_nzd[nSel] = nz_dims;
	if(nz_dims > 0)
	{
		c_idx[nSel] = new uint32_t[nz_dims];
		ASSERT(c_idx[nSel]);
		memset(c_idx[nSel], 0, sizeof(uint32_t)*nz_dims);
		c_val[nSel] = new double[nz_dims];
		memset(c_val[nSel], 0, sizeof(double)*nz_dims);
		ASSERT(c_val[nSel]);

		uint32_t idx=0;
		for(j=0; j < nDim; j++ )
		{
			if(new_a[j] != 0)
			{
				c_idx[nSel][idx] = j;
				c_val[nSel][idx++] = new_a[j];
			}
		}
	}

	new_col_H[nSel] = sq_norm_a;
	for(i=0; i < nSel; i++)
	{
		double tmp = 0;
		for(j=0; j < c_nzd[i]; j++)
			tmp += new_a[c_idx[i][j]]*c_val[i][j];

		new_col_H[i] = tmp;
	}
}

void CSVMOcas::sort( double* vals, uint32_t* idx, uint32_t size)
{
	CMath::qsort_index(vals, idx, size);
}

/*----------------------------------------------------------------------
  sparse_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
void CSVMOcas::compute_output( double *output, void* ptr )
{
	CSVMOcas* o = (CSVMOcas*) ptr;
	CSparseFeatures<DREAL>* f=o->get_features();
	INT nData=f->get_num_vectors();

	DREAL* y = o->lab;
//	f->dense_dot_range(output, 0, nData, y, o->w, o->w_dim, 0.0);

	for (INT i=0; i<nData; i++)
		output[i]=y[i]*f->dense_dot(1.0, i, o->w, o->w_dim, 0.0);

}

/*----------------------------------------------------------------------
  sq_norm_W = compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = sparse_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void CSVMOcas::compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* ptr )
{
	CSVMOcas* o = (CSVMOcas*) ptr;
	uint32_t nDim= (uint32_t) o->w_dim;
	//CMath::swap(o->w, o->old_w);
	double* W=o->w;
	double* oldW=o->old_w;
	memcpy(oldW, W, sizeof(double)*nDim ); 
	//memset(W, 0, sizeof(double)*nDim);

	DREAL** c_val = o->cp_value;
	uint32_t** c_idx = o->cp_index;
	uint32_t* c_nzd = o->cp_nz_dims;

	memset(W, 0, sizeof(double)*nDim);

	for(uint32_t i=0; i<nSel; i++)
	{
		uint32_t nz_dims = c_nzd[i];

		if(nz_dims > 0 && alpha[i] > 0)
		{
			for(uint32_t j=0; j < nz_dims; j++)
				W[c_idx[i][j]] += alpha[i]*c_val[i][j];
		}
	}

	*sq_norm_W = CMath::dot(W,W, nDim);
	*dp_WoldW = CMath::dot(W,oldW, nDim);;
}

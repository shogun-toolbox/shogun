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

CSVMOcas::CSVMOcas(E_SVM_TYPE type) : CSparseLinearClassifier(), C1(1), C2(1),
	epsilon(1e-3), method(type)
{
}

CSVMOcas::CSVMOcas(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
: CSparseLinearClassifier(), C1(C), C2(C), epsilon(1e-3)
{
	method=SVM_OCAS;
	CSparseLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSVMOcas::~CSVMOcas()
{
}

bool CSVMOcas::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());

	INT num_train_labels=get_labels()->get_num_labels();
	//INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);

	//delete[] w;
	//w=new DREAL[num_feat];
	//ASSERT(w);
//bias=0;
//
	double TolAbs=0;
	double QPBound=0;
	int BufSize=3000;
	int Method=0;
	ocas_return_value_T result = svm_ocas_solver( get_C1(), num_vec, get_epsilon(),
			TolAbs, QPBound, BufSize, Method, 
			&CSVMOcas::sparse_compute_W,
			&CSVMOcas::sparse_update_W, 
			&CSVMOcas::sparse_add_new_cut, 
			&CSVMOcas::sparse_compute_output,
			&CMath::qsort_index<double,uint32_t>);
	return true;
}

/*----------------------------------------------------------------------
  in-place computes sparse_mat(:,col)= alpha * sparse_mat(:,col)
  where alpha is a scalar and sparse_mat is Matlab sparse matrix.
  ----------------------------------------------------------------------*/
void CSVMOcas::mul_sparse_col(double alpha, CSparseFeatures<DREAL>* sparse_mat, uint32_t col)
{
//	uint32_t nItems, ptr, i, row;
//	mwSize *Ir, *Jc;
//	double *Pr, val;
//
//	Ir = mxGetIr(sparse_mat);
//	Jc = mxGetJc(sparse_mat);
//	Pr = mxGetPr(sparse_mat);
//
//	nItems = Jc[col+1] - Jc[col];
//	ptr = Jc[col];
//
//	for(i=0; i < nItems; i++)
//		Pr[ptr++]*=alpha;
}


/*----------------------------------------------------------------------
 It computes full_vec = full_vec + sparse_mat(:,col)
 where full_vec is a double array and sparse_mat is Matlab 
 sparse matrix.
  ----------------------------------------------------------------------*/
void CSVMOcas::add_sparse_col(double *full_vec, CSparseFeatures<DREAL>* sparse_mat, uint32_t col)
{
//  uint32_t nItems, ptr, i, row;
//  mwSize *Ir, *Jc;
//  double *Pr, val;
//    
//  Ir = mxGetIr(sparse_mat);
//  Jc = mxGetJc(sparse_mat);
//  Pr = mxGetPr(sparse_mat);
//
//  nItems = Jc[col+1] - Jc[col];
//  ptr = Jc[col];
//
//  for(i=0; i < nItems; i++) {
//    val = Pr[ptr];
//    row = Ir[ptr++];
//
//    full_vec[row] += val;
//  }
}

/*----------------------------------------------------------------------
 It computes dp = full_vec'*sparse_mat(:,col)
 where full_vec is a double array and sparse_mat is Matlab 
 sparse matrix.
  ----------------------------------------------------------------------*/
double CSVMOcas::dp_sparse_col(double *full_vec, CSparseFeatures<DREAL>* sparse_mat, uint32_t col)
{
//  uint32_t nItems, ptr, i, row;
//  mwSize *Ir, *Jc;
//  double *Pr, val, dp;
//
//  Ir = mxGetIr(sparse_mat);
//  Jc = mxGetJc(sparse_mat);
//  Pr = mxGetPr(sparse_mat);
//
//  dp = 0;
//  nItems = Jc[col+1] - Jc[col];
//  ptr = Jc[col];
//
//  for(i=0; i < nItems; i++) {
//    val = Pr[ptr];
//    row = Ir[ptr++];
//
//    dp += full_vec[row]*val;
//  }
//
//  return(dp);  
return 0.0;
}


/*----------------------------------------------------------------------------------
  sq_norm_W = sparse_update_W( t ) does the following:

  W = oldW*(1-t) + t*W;
  sq_norm_W = W'*W;

  ---------------------------------------------------------------------------------*/
double CSVMOcas::sparse_update_W( double t )
{
//  uint32_t j;
//  double sq_norm_W = 0;         
//
//
//  for(j=0; j <nDim; j++) {
//    W[j] = oldW[j]*(1-t) + t*W[j];
//    sq_norm_W += W[j]*W[j];
//  }          
//
//  return( sq_norm_W );
	return 0;
}

/*----------------------------------------------------------------------------------
  sparse_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [sparse_A(:,1:nSel)'*new_a ; new_a'*new_a];
    sparse_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
void CSVMOcas::sparse_add_new_cut( double *new_col_H, 
                  uint32_t *new_cut, 
                  uint32_t cut_length, 
                  uint32_t nSel )
{
//  double *new_a, sq_norm_a;
//  uint32_t i, j, nz_dims, ptr;
//
//  /* temporary vector */
//  new_a = (double*)mxCalloc(nDim,sizeof(double));
//  if(new_a == NULL) mexErrMsgTxt("Not enough memory for vector new_a.");
//  
//  for(i=0; i < cut_length; i++) 
//    add_sparse_col(new_a, data_X, new_cut[i]);
// 
//  /* compute new_a'*new_a and count number of non-zerou dimensions */
//  nz_dims = 0; 
//  sq_norm_a = 0;
//  for(j=0; j < nDim; j++ ) {
//    if(new_a[j] != 0) {
//      nz_dims++;
//      sq_norm_a += new_a[j]*new_a[j];
//    }
//  }
//
//  /* sparsify new_a and insert it to the last column  of sparse_A */
//  sparse_A.nz_dims[nSel] = nz_dims;
//  if(nz_dims > 0) {
//    sparse_A.index[nSel] = mxCalloc(nz_dims,sizeof(uint32_t));
//    sparse_A.value[nSel] = mxCalloc(nz_dims,sizeof(double));
//    if(sparse_A.index[nSel]==NULL || sparse_A.value[nSel]==NULL)
//      mexErrMsgTxt("Not enough memory for vector sparse_A.index[nSel], sparse_A.value[nSel].");
//
//    ptr = 0;
//    for(j=0; j < nDim; j++ ) {
//      if(new_a[j] != 0) {
//        sparse_A.index[nSel][ptr] = j;
//        sparse_A.value[nSel][ptr++] = new_a[j];
//      }
//    }
//  }
//   
//  new_col_H[nSel] = sq_norm_a;
//  for(i=0; i < nSel; i++) {
//    double tmp = 0;
//    for(j=0; j < sparse_A.nz_dims[i]; j++) {
//      tmp += new_a[sparse_A.index[i][j]]*sparse_A.value[i][j];
//    }
//      
//    new_col_H[i] = tmp;
//  }
//
//  mxFree( new_a );
//  return;
}

/*----------------------------------------------------------------------
  sparse_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
void CSVMOcas::sparse_compute_output( double *output )
{
//  uint32_t i;
//
//  for(i=0; i < nData; i++) { 
//    output[i] = dp_sparse_col(W, data_X, i);
//  }
//  
//  return;
}

/*----------------------------------------------------------------------
  sq_norm_W = sparse_compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = sparse_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void CSVMOcas::sparse_compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel )
{
//  uint32_t i,j, nz_dims;
//
//  memcpy(oldW, W, sizeof(double)*nDim ); 
//  memset(W, 0, sizeof(double)*nDim);
//
//  for(i=0; i < nSel; i++) {
//    nz_dims = sparse_A.nz_dims[i];
//    if(nz_dims > 0 && alpha[i] > 0) {
//      for(j=0; j < nz_dims; j++) {
//        W[sparse_A.index[i][j]] += alpha[i]*sparse_A.value[i][j];
//      }
//    }
//  }
//
//  *sq_norm_W = 0;
//  *dp_WoldW = 0;
//  for(j=0; j < nDim; j++) {
//    *sq_norm_W += W[j]*W[j];
//    *dp_WoldW += W[j]*oldW[j];
//  }
//  
//  return;
}


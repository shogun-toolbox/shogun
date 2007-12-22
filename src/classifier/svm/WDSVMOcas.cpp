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
#include "lib/DynamicArray.h"
#include "lib/Time.h"
#include "base/Parallel.h"
#include "classifier/Classifier.h"
#include "classifier/svm/libocas.h"
#include "classifier/svm/WDSVMOcas.h"
#include "features/StringFeatures.h"
#include "features/Alphabet.h"
#include "features/Labels.h"

CWDSVMOcas::CWDSVMOcas(E_SVM_TYPE type) : CClassifier(), use_bias(false), bufsize(3000), C1(1), C2(1),
	epsilon(1e-3), method(type)
{
	w=NULL;
	old_w=NULL;
	degree=6;
	from_degree=40;
	wd_weights=NULL;
	w_offsets=NULL;
	normalization_const=1.0;
}

CWDSVMOcas::CWDSVMOcas(DREAL C, INT d, INT from_d, CStringFeatures<BYTE>* traindat, CLabels* trainlab) 
: CClassifier(), use_bias(false), bufsize(3000), C1(C), C2(C), epsilon(1e-3), degree(d),
	from_degree(from_d)
{
	w=NULL;
	old_w=NULL;
	method=SVM_OCAS;
	features=traindat;
	CClassifier::labels=trainlab;
	wd_weights=NULL;
	w_offsets=NULL;
	normalization_const=1.0;
}


CWDSVMOcas::~CWDSVMOcas()
{
}

CLabels* CWDSVMOcas::classify(CLabels* output)
{
	set_wd_weights();
	set_normalization_const();

	if (features)
	{
		INT num=features->get_num_vectors();
		ASSERT(num>0);

		if (!output)
			output=new CLabels(num);

		ASSERT(output);
		for (INT i=0; i<num; i++)
			output->set_label(i, classify_example(i));

		return output;
	}

	return NULL;
}

INT CWDSVMOcas::set_wd_weights()
{
	ASSERT(degree>0 && degree<8);
	delete[] wd_weights;
	wd_weights=new SHORTREAL[degree];
	ASSERT(wd_weights);
	delete[] w_offsets;
	w_offsets=new INT[degree];
	ASSERT(w_offsets);
	INT w_dim_single_char=0;

	for (INT i=0; i<degree; i++)
	{
		w_offsets[i]=CMath::pow(alphabet_size, i+1);
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));
		w_dim_single_char+=w_offsets[i];
	}
	return w_dim_single_char;
}

bool CWDSVMOcas::train()
{
	SG_INFO("C=%f, epsilon=%f, bufsize=%d\n", get_C1(), get_epsilon(), bufsize);

	ASSERT(get_labels());
	ASSERT(get_features());
	ASSERT(get_labels()->is_two_class_labeling());
	CAlphabet* alphabet=get_features()->get_alphabet();
	ASSERT(alphabet && alphabet->get_alphabet()==RAWDNA);

	alphabet_size=alphabet->get_num_symbols();
	string_length=features->get_num_vectors();
	INT num_train_labels=0;
	lab=get_labels()->get_labels(num_train_labels);

	INT w_dim_single_char=set_wd_weights();
	CMath::display_vector(wd_weights, degree, "wd_weights");
	SG_DEBUG("w_dim_single_char=%d\n", w_dim_single_char);
	w_dim=string_length*w_dim_single_char;
	SG_DEBUG("cutting plane has %d dims\n", w_dim);
	num_vec=get_features()->get_max_vector_length();

	set_normalization_const();
	ASSERT(num_vec==num_train_labels);
	ASSERT(num_vec>0);


	delete[] w;
	w=new SHORTREAL[w_dim];
	ASSERT(w);
	memset(w, 0, w_dim*sizeof(SHORTREAL));

	delete[] old_w;
	old_w=new SHORTREAL[w_dim];
	ASSERT(old_w);
	memset(old_w, 0, w_dim*sizeof(SHORTREAL));
	bias=0;

	cuts=new SHORTREAL*[bufsize];
	ASSERT(cuts);
	memset(cuts, 0, sizeof(*cuts)*bufsize);

/////speed tests/////
	/*double* tmp = new double[num_vec];
	ASSERT(tmp);
	double start=CTime::get_curtime();
	CMath::random_vector(w, w_dim, (SHORTREAL) 0, (SHORTREAL) 1000);
	compute_output(tmp, this);
	start=CTime::get_curtime()-start;
	SG_PRINT("timing:%f\n", start);
	delete[] tmp;
	exit(1);*/
/////speed tests/////
	double TolAbs=0;
	double QPBound=0;
	int Method=0;
	if (method == SVM_OCAS)
		Method = 1;
	ocas_return_value_T result = svm_ocas_solver( get_C1(), num_vec, get_epsilon(),
			TolAbs, QPBound, bufsize, Method, 
			&CWDSVMOcas::compute_W,
			&CWDSVMOcas::update_W, 
			&CWDSVMOcas::add_new_cut, 
			&CWDSVMOcas::compute_output,
			&CWDSVMOcas::sort,
			&printf,
			this);

	SG_INFO("Ocas Converged after %d iterations\n"
			"==================================\n"
			"timing statistics:\n"
			"output_time: %f s\n"
			"sort_time: %f s\n"
			"add_time: %f s\n"
			"w_time: %f s\n"
			"solver_time %f s\n"
			"ocas_time %f s\n\n", result.nIter, result.output_time, result.sort_time,
			result.add_time, result.w_time, result.solver_time, result.ocas_time);

	for (INT i=bufsize-1; i>=0; i--)
		delete[] cuts[i];
	delete[] cuts;

	delete[] lab;
	lab=NULL;

	return true;
}

/*----------------------------------------------------------------------------------
  sq_norm_W = sparse_update_W( t ) does the following:

  W = oldW*(1-t) + t*W;
  sq_norm_W = W'*W;

  ---------------------------------------------------------------------------------*/
double CWDSVMOcas::update_W( double t, void* ptr )
{
  double sq_norm_W = 0;         
  CWDSVMOcas* o = (CWDSVMOcas*) ptr;
  uint32_t nDim = (uint32_t) o->w_dim;
  SHORTREAL* W=o->w;
  SHORTREAL* oldW=o->old_w;

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
void CWDSVMOcas::add_new_cut( double *new_col_H, 
                  uint32_t *new_cut, 
                  uint32_t cut_length, 
                  uint32_t nSel,
				  void* ptr)
{
	CWDSVMOcas* o = (CWDSVMOcas*) ptr;
	CStringFeatures<BYTE>* f = o->features;
	INT string_length = o->string_length;
	uint32_t nDim=(uint32_t) o->w_dim;
	DREAL* y = o->lab;
	INT alphabet_size = o->alphabet_size;
	INT* w_offsets = o->w_offsets;
	SHORTREAL* wd_weights = o->wd_weights;
	INT degree = o->degree;
	SHORTREAL** cuts=o->cuts;
	DREAL normalization_const = o->normalization_const;

	uint32_t i;

	// temporary vector
	SHORTREAL* new_a = new SHORTREAL[nDim];
	memset(new_a, 0, sizeof(SHORTREAL)*nDim);

	INT* val=new INT[cut_length];
	ASSERT(val);

	INT offs=0;
	for (INT j=0; j<string_length; j++)
	{
		memset(val,0,sizeof(INT)*cut_length);
		INT lim=CMath::min(degree, string_length-j);
		INT len;

		for (INT k=0; k<lim; k++)
		{
			BYTE* vec = f->get_feature_vector(j+k, len);
			SHORTREAL wd = wd_weights[k];

			for(i=0; i < cut_length; i++) 
			{
				val[i]=val[i]*alphabet_size + vec[new_cut[i]];
				new_a[offs+val[i]]+=wd * y[new_cut[i]]/normalization_const;
			}
			offs+=w_offsets[k];
		}
	}

	// insert new_a into the last column of sparse_A
	for(i=0; i < nSel; i++)
		new_col_H[i] = CMath::dot(new_a, cuts[i], nDim);
	new_col_H[nSel] = CMath::dot(new_a, new_a, nDim);

	cuts[nSel]=new_a;
	//CMath::display_vector(new_col_H, nSel+1, "new_col_H");
	//CMath::display_vector(cuts[nSel], nDim, "cut[nSel]");
	delete[] val;
}

void CWDSVMOcas::sort( double* vals, uint32_t* idx, uint32_t size)
{
	CMath::qsort_index(vals, idx, size);
}

/*----------------------------------------------------------------------
  sparse_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
void CWDSVMOcas::compute_output( double *output, void* ptr )
{
	CWDSVMOcas* o = (CWDSVMOcas*) ptr;
	CStringFeatures<BYTE>* f=o->get_features();
	INT nData=o->num_vec;

	INT degree = o->degree;
	INT string_length = o->string_length;
	INT alphabet_size = o->alphabet_size;
	INT* w_offsets = o->w_offsets;
	SHORTREAL* wd_weights = o->wd_weights;
	SHORTREAL* w= o->w;

	DREAL* y = o->lab;
	DREAL normalization_const = o->normalization_const;

	SHORTREAL* out=new SHORTREAL[nData];
	ASSERT(out);
	INT* val=new INT[nData];
	ASSERT(val);
	memset(out, 0, sizeof(SHORTREAL)*nData);

	INT offs=0;
	for (INT j=0; j<string_length; j++)
	{
		memset(val,0,sizeof(INT)*nData);
		INT lim=CMath::min(degree, string_length-j);
		INT len;

		for (INT k=0; k<lim; k++)
		{
			BYTE* vec=f->get_feature_vector(j+k, len);
			SHORTREAL wd = wd_weights[k];

			for (INT i=0; i<nData; i++) // quite fast 1.9s
			{
				val[i]=val[i]*alphabet_size + vec[i];
				output[i]+=wd*w[offs+val[i]];
			}

			/*for (INT i=0; i<nData/4; i++) // slowest 2s
			{
				UINT x=((UINT*) vec)[i];
				INT ii=4*i;
				val[ii]=val[ii]*alphabet_size + (x&255);
				val[ii+1]=val[ii+1]*alphabet_size + ((x>>8)&255);
				val[ii+2]=val[ii+2]*alphabet_size + ((x>>16)&255);
				val[ii+3]=val[ii+3]*alphabet_size + (x>>24);
				output[ii]+=wd*w[offs+val[ii]];
				output[ii+1]+=wd*w[offs+val[ii+1]];
				output[ii+2]+=wd*w[offs+val[ii+2]];
				output[ii+3]+=wd*w[offs+val[ii+3]];
			}*/

			/*for (INT i=0; i<nData>>3; i++) // fastest on 64bit: 1.5s
			{
				ULONG x=((ULONG*) vec)[i];
				INT ii=i<<3;
				val[ii]=val[ii]*alphabet_size + (x&255);
				val[ii+1]=val[ii+1]*alphabet_size + ((x>>8)&255);
				val[ii+2]=val[ii+2]*alphabet_size + ((x>>16)&255);
				val[ii+3]=val[ii+3]*alphabet_size + ((x>>24)&255);
				val[ii+4]=val[ii+4]*alphabet_size + ((x>>32)&255);
				val[ii+5]=val[ii+5]*alphabet_size + ((x>>40)&255);
				val[ii+6]=val[ii+6]*alphabet_size + ((x>>48)&255);
				val[ii+7]=val[ii+7]*alphabet_size + (x>>56);
				out[ii]+=wd*w[offs+val[ii]];
				out[ii+1]+=wd*w[offs+val[ii+1]];
				out[ii+2]+=wd*w[offs+val[ii+2]];
				out[ii+3]+=wd*w[offs+val[ii+3]];
				out[ii+4]+=wd*w[offs+val[ii+4]];
				out[ii+5]+=wd*w[offs+val[ii+5]];
				out[ii+6]+=wd*w[offs+val[ii+6]];
				out[ii+7]+=wd*w[offs+val[ii+7]];
			}*/
			offs+=w_offsets[k];
		}
	}

	for (INT i=0; i<nData; i++)
		output[i]=out[i]*y[i]/normalization_const;

	delete[] val;
	delete[] out;
}
/*----------------------------------------------------------------------
  sq_norm_W = compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = sparse_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void CWDSVMOcas::compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* ptr )
{
	CWDSVMOcas* o = (CWDSVMOcas*) ptr;
	uint32_t nDim= (uint32_t) o->w_dim;
	CMath::swap(o->w, o->old_w);
	SHORTREAL* W=o->w;
	SHORTREAL* oldW=o->old_w;
	SHORTREAL** cuts=o->cuts;
	memset(W, 0, sizeof(SHORTREAL)*nDim);

	for (uint32_t i=0; i<nSel; i++)
	{
		if (alpha[i] > 0)
			CMath::vec1_plus_scalar_times_vec2(W, alpha[i], cuts[i], nDim);
	}

	*sq_norm_W = CMath::dot(W,W, nDim);
	*dp_WoldW = CMath::dot(W,oldW, nDim);;
	//SG_PRINT("nSel=%d sq_norm_W=%f dp_WoldW=%f\n", nSel, *sq_norm_W, *dp_WoldW);
}

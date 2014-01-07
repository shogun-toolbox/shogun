/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Vojtech Franc
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <labels/Labels.h>
#include <mathematics/Math.h>
#include <lib/Time.h>
#include <base/Parameter.h>
#include <base/Parallel.h>
#include <machine/LinearMachine.h>
#include <classifier/svm/SVMOcas.h>
#include <features/DotFeatures.h>
#include <labels/Labels.h>
#include <labels/BinaryLabels.h>

using namespace shogun;

CSVMOcas::CSVMOcas()
: CLinearMachine()
{
	init();
}

CSVMOcas::CSVMOcas(E_SVM_TYPE type)
: CLinearMachine()
{
	init();
	method=type;
}

CSVMOcas::CSVMOcas(
	float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine()
{
	init();
	C1=C;
	C2=C;

	set_features(traindat);
	set_labels(trainlab);
}


CSVMOcas::~CSVMOcas()
{
}

bool CSVMOcas::train_machine(CFeatures* data)
{
	SG_INFO("C=%f, epsilon=%f, bufsize=%d\n", get_C1(), get_epsilon(), bufsize)
	SG_DEBUG("use_bias = %i\n", get_bias_enabled())

	ASSERT(m_labels)
  ASSERT(m_labels->get_label_type() == LT_BINARY)
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}
	ASSERT(features)

	int32_t num_vec=features->get_num_vectors();
	lab = SGVector<float64_t>(num_vec);
	for (int32_t i=0; i<num_vec; i++)
		lab[i] = ((CBinaryLabels*)m_labels)->get_label(i);

	w=SGVector<float64_t>(features->get_dim_feature_space());
	w.zero();

	if (num_vec!=lab.vlen || num_vec<=0)
		SG_ERROR("num_vec=%d num_train_labels=%d\n", num_vec, lab.vlen)

	SG_FREE(old_w);
	old_w=SG_CALLOC(float64_t, w.vlen);
	bias=0;
	old_bias=0;

	tmp_a_buf=SG_CALLOC(float64_t, w.vlen);
	cp_value=SG_CALLOC(float64_t*, bufsize);
	cp_index=SG_CALLOC(uint32_t*, bufsize);
	cp_nz_dims=SG_CALLOC(uint32_t, bufsize);
	cp_bias=SG_CALLOC(float64_t, bufsize);

	float64_t TolAbs=0;
	float64_t QPBound=0;
	int32_t Method=0;
	if (method == SVM_OCAS)
		Method = 1;
	ocas_return_value_T result = svm_ocas_solver( get_C1(), num_vec, get_epsilon(),
			TolAbs, QPBound, get_max_train_time(), bufsize, Method,
			&CSVMOcas::compute_W,
			&CSVMOcas::update_W,
			&CSVMOcas::add_new_cut,
			&CSVMOcas::compute_output,
			&CSVMOcas::sort,
			&CSVMOcas::print,
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
			result.add_time, result.w_time, result.qp_solver_time, result.ocas_time);

	SG_FREE(tmp_a_buf);

	primal_objective = result.Q_P;

	uint32_t num_cut_planes = result.nCutPlanes;

	SG_DEBUG("num_cut_planes=%d\n", num_cut_planes)
	for (uint32_t i=0; i<num_cut_planes; i++)
	{
		SG_DEBUG("cp_value[%d]=%p\n", i, cp_value)
		SG_FREE(cp_value[i]);
		SG_DEBUG("cp_index[%d]=%p\n", i, cp_index)
		SG_FREE(cp_index[i]);
	}

	SG_FREE(cp_value);
	cp_value=NULL;
	SG_FREE(cp_index);
	cp_index=NULL;
	SG_FREE(cp_nz_dims);
	cp_nz_dims=NULL;
	SG_FREE(cp_bias);
	cp_bias=NULL;

	SG_FREE(old_w);
	old_w=NULL;

	return true;
}

/*----------------------------------------------------------------------------------
  sq_norm_W = sparse_update_W( t ) does the following:

  W = oldW*(1-t) + t*W;
  sq_norm_W = W'*W;

  ---------------------------------------------------------------------------------*/
float64_t CSVMOcas::update_W( float64_t t, void* ptr )
{
  float64_t sq_norm_W = 0;
  CSVMOcas* o = (CSVMOcas*) ptr;
  uint32_t nDim = (uint32_t) o->w.vlen;
  float64_t* W=o->w.vector;
  float64_t* oldW=o->old_w;

  for(uint32_t j=0; j <nDim; j++)
  {
	  W[j] = oldW[j]*(1-t) + t*W[j];
	  sq_norm_W += W[j]*W[j];
  }
  o->bias=o->old_bias*(1-t) + t*o->bias;
  sq_norm_W += CMath::sq(o->bias);

  return( sq_norm_W );
}

/*----------------------------------------------------------------------------------
  sparse_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [sparse_A(:,1:nSel)'*new_a ; new_a'*new_a];
    sparse_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
int CSVMOcas::add_new_cut(
	float64_t *new_col_H, uint32_t *new_cut, uint32_t cut_length,
	uint32_t nSel, void* ptr)
{
	CSVMOcas* o = (CSVMOcas*) ptr;
	CDotFeatures* f = o->features;
	uint32_t nDim=(uint32_t) o->w.vlen;
	float64_t* y = o->lab.vector;

	float64_t** c_val = o->cp_value;
	uint32_t** c_idx = o->cp_index;
	uint32_t* c_nzd = o->cp_nz_dims;
	float64_t* c_bias = o->cp_bias;

	float64_t sq_norm_a;
	uint32_t i, j, nz_dims;

	/* temporary vector */
	float64_t* new_a = o->tmp_a_buf;
	memset(new_a, 0, sizeof(float64_t)*nDim);

	for(i=0; i < cut_length; i++)
	{
		f->add_to_dense_vec(y[new_cut[i]], new_cut[i], new_a, nDim);

		if (o->use_bias)
			c_bias[nSel]+=y[new_cut[i]];
	}

	/* compute new_a'*new_a and count number of non-zerou dimensions */
	nz_dims = 0;
	sq_norm_a = CMath::sq(c_bias[nSel]);
	for(j=0; j < nDim; j++ ) {
		if(new_a[j] != 0) {
			nz_dims++;
			sq_norm_a += new_a[j]*new_a[j];
		}
	}

	/* sparsify new_a and insert it to the last column of sparse_A */
	c_nzd[nSel] = nz_dims;
	c_idx[nSel]=NULL;
	c_val[nSel]=NULL;

	if(nz_dims > 0)
	{
		c_idx[nSel]=SG_MALLOC(uint32_t, nz_dims);
		c_val[nSel]=SG_MALLOC(float64_t, nz_dims);

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
		float64_t tmp = c_bias[nSel]*c_bias[i];
		for(j=0; j < c_nzd[i]; j++)
			tmp += new_a[c_idx[i][j]]*c_val[i][j];

		new_col_H[i] = tmp;
	}
	//CMath::display_vector(new_col_H, nSel+1, "new_col_H");
	//CMath::display_vector((int32_t*) c_idx[nSel], (int32_t) nz_dims, "c_idx");
	//CMath::display_vector((float64_t*) c_val[nSel], nz_dims, "c_val");
	return 0;
}

int CSVMOcas::sort(float64_t* vals, float64_t* data, uint32_t size)
{
	CMath::qsort_index(vals, data, size);
	return 0;
}

/*----------------------------------------------------------------------
  sparse_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
int CSVMOcas::compute_output(float64_t *output, void* ptr)
{
	CSVMOcas* o = (CSVMOcas*) ptr;
	CDotFeatures* f=o->features;
	int32_t nData=f->get_num_vectors();

	float64_t* y = o->lab.vector;

	f->dense_dot_range(output, 0, nData, y, o->w.vector, o->w.vlen, 0.0);

	for (int32_t i=0; i<nData; i++)
		output[i]+=y[i]*o->bias;
	//CMath::display_vector(o->w, o->w.vlen, "w");
	//CMath::display_vector(output, nData, "out");
	return 0;
}

/*----------------------------------------------------------------------
  sq_norm_W = compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = sparse_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void CSVMOcas::compute_W(
	float64_t *sq_norm_W, float64_t *dp_WoldW, float64_t *alpha, uint32_t nSel,
	void* ptr )
{
	CSVMOcas* o = (CSVMOcas*) ptr;
	uint32_t nDim= (uint32_t) o->w.vlen;
	CMath::swap(o->w.vector, o->old_w);
	float64_t* W=o->w.vector;
	float64_t* oldW=o->old_w;
	memset(W, 0, sizeof(float64_t)*nDim);
	float64_t old_bias=o->bias;
	float64_t bias=0;

	float64_t** c_val = o->cp_value;
	uint32_t** c_idx = o->cp_index;
	uint32_t* c_nzd = o->cp_nz_dims;
	float64_t* c_bias = o->cp_bias;

	for(uint32_t i=0; i<nSel; i++)
	{
		uint32_t nz_dims = c_nzd[i];

		if(nz_dims > 0 && alpha[i] > 0)
		{
			for(uint32_t j=0; j < nz_dims; j++)
				W[c_idx[i][j]] += alpha[i]*c_val[i][j];
		}
		bias += c_bias[i]*alpha[i];
	}

	*sq_norm_W = SGVector<float64_t>::dot(W,W, nDim) + CMath::sq(bias);
	*dp_WoldW = SGVector<float64_t>::dot(W,oldW, nDim) + bias*old_bias;
	//SG_PRINT("nSel=%d sq_norm_W=%f dp_WoldW=%f\n", nSel, *sq_norm_W, *dp_WoldW)

	o->bias = bias;
	o->old_bias = old_bias;
}

void CSVMOcas::init()
{
	use_bias=true;
	bufsize=3000;
	C1=1;
	C2=1;

	epsilon=1e-3;
	method=SVM_OCAS;
	old_w=NULL;
	tmp_a_buf=NULL;
	cp_value=NULL;
	cp_index=NULL;
	cp_nz_dims=NULL;
	cp_bias=NULL;

	primal_objective = 0.0;

    m_parameters->add(&C1, "C1",  "Cost constant 1.");
    m_parameters->add(&C2, "C2",  "Cost constant 2.");
    m_parameters->add(&use_bias, "use_bias",
			"Indicates if bias is used.");
    m_parameters->add(&epsilon, "epsilon", "Convergence precision.");
    m_parameters->add(&bufsize, "bufsize", "Maximum number of cutting planes.");
    m_parameters->add((machine_int_t*) &method, "method",
			"SVMOcas solver type.");
}

float64_t CSVMOcas::compute_primal_objective() const
{
	return primal_objective;
}


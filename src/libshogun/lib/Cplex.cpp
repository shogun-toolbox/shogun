/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef USE_CPLEX
#include <unistd.h>

#include "lib/Cplex.h"
#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Mathematics.h"

using namespace shogun;

CCplex::CCplex()
: CSGObject(), env(NULL), lp(NULL), lp_initialized(false)
{
}

CCplex::~CCplex()
{
	cleanup();
}

bool CCplex::init(E_PROB_TYPE typ, int32_t timeout)
{
	problem_type=typ;

	while (env==NULL)
	{
		int status = 1; /* for calling external lib */
		env = CPXopenCPLEX (&status);

		if ( env == NULL )
		{
			char  errmsg[1024];
			SG_WARNING("Could not open CPLEX environment.\n");
			CPXgeterrorstring (env, status, errmsg);
			SG_WARNING("%s", errmsg);
			SG_WARNING("retrying in %d seconds\n", timeout);
			sleep(timeout);
		}
		else
		{
			/* Turn on output to the screen */

			status = CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_OFF);
			if (status)
				SG_ERROR( "Failure to turn off screen indicator, error %d.\n", status);

			{
				status = CPXsetintparam (env, CPX_PARAM_DATACHECK, CPX_ON);
				if (status)
					SG_ERROR( "Failure to turn on data checking, error %d.\n", status);
				else
				{
					lp = CPXcreateprob (env, &status, "shogun");

					if ( lp == NULL )
						SG_ERROR( "Failed to create optimization problem.\n");
					else
						CPXchgobjsen (env, lp, CPX_MIN);  /* Problem is minimization */

					if (problem_type==E_QP)
						status = CPXsetintparam (env, CPX_PARAM_QPMETHOD, 0);
					else if (problem_type == E_LINEAR)
						status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, 0);
					if (status)
						SG_ERROR( "Failure to select dual lp/qp optimization, error %d.\n", status);

				}
			}
		}
	}

	return (lp != NULL) && (env != NULL);
}

bool CCplex::setup_subgradientlpm_QP(
	float64_t C, CLabels* labels, CSparseFeatures<float64_t>* features,
	int32_t* idx_bound, int32_t num_bound, int32_t* w_zero, int32_t num_zero,
	float64_t* vee, int32_t num_dim, bool use_bias)
{
	const int cmatsize=10*1024*1024; //no more than 10mil. elements
	bool result=false;
	int32_t num_variables=num_dim + num_bound+num_zero; // xi, beta

	ASSERT(num_dim>0);
	ASSERT(num_dim>0);

	// setup LP part
	float64_t* lb=new float64_t[num_variables];
	float64_t* ub=new float64_t[num_variables];
	float64_t* obj=new float64_t[num_variables];
	char* sense = new char[num_dim];
	int* cmatbeg=new int[num_variables];
	int* cmatcnt=new int[num_variables];
	int* cmatind=new int[cmatsize];
	double* cmatval=new double[cmatsize];

	for (int32_t i=0; i<num_variables; i++)
	{
		obj[i]=0;

		if (i<num_dim) //xi part
		{
			lb[i]=-CPX_INFBOUND;
			ub[i]=+CPX_INFBOUND;
		}
		else //beta part
		{
			lb[i]=0.0;
			ub[i]=1.0;
		}
	}

	int32_t offs=0;
	for (int32_t i=0; i<num_variables; i++)
	{
		if (i<num_dim) //sum_xi
		{
			sense[i]='E';
			cmatbeg[i]=offs;
			cmatcnt[i]=1;
			cmatind[offs]=offs;
			cmatval[offs]=1.0;
			offs++;
			ASSERT(offs<cmatsize);
		}
		else if (i<num_dim+num_zero) // Z_w*beta_w
		{
			cmatbeg[i]=offs;
			cmatcnt[i]=1;
			cmatind[offs]=w_zero[i-num_dim];
			cmatval[offs]=-1.0;
			offs++;
			ASSERT(offs<cmatsize);
		}
		else // Z_x*beta_x
		{
			int32_t idx=idx_bound[i-num_dim-num_zero];
			int32_t vlen=0;
			bool vfree=false;
			//SG_PRINT("idx=%d\n", idx);
			SGSparseMatrixEntry<float64_t>* vec=features->get_sparse_feature_vector(idx, vlen, vfree);
			//SG_PRINT("vlen=%d\n", vlen);

			cmatbeg[i]=offs;
			cmatcnt[i]=vlen;

			float64_t val= -C*labels->get_label(idx);

			if (vlen>0)
			{
				for (int32_t j=0; j<vlen; j++)
				{
					cmatind[offs]=vec[j].feat_index;
					cmatval[offs]=-val*vec[j].entry;
					offs++;
					ASSERT(offs<cmatsize);
					//SG_PRINT("vec[%d]=%10.10f\n", j, vec[j].entry);
				}

				if (use_bias)
				{
					cmatcnt[i]++;
					cmatind[offs]=num_dim-1;
					cmatval[offs]=-val;
					offs++;
					ASSERT(offs<cmatsize);
				}
			}
			else
			{
				if (use_bias)
				{
					cmatcnt[i]++;
					cmatind[offs]=num_dim-1;
					cmatval[offs]=-val;
				}
				else
					cmatval[offs]=0.0;
				offs++;
				ASSERT(offs<cmatsize);
			}

			features->free_feature_vector(vec, idx, vfree);
		}
	}

	result = CPXcopylp(env, lp, num_variables, num_dim, CPX_MIN, 
			obj, vee, sense, cmatbeg, cmatcnt, cmatind, cmatval, lb, ub, NULL) == 0;

	if (!result)
		SG_ERROR("CPXcopylp failed.\n");

	//write_problem("problem.lp");

	delete[] sense;
	delete[] lb;
	delete[] ub;
	delete[] obj;
	delete[] cmatbeg;
	delete[] cmatcnt;
	delete[] cmatind;
	delete[] cmatval;

	//// setup QP part (diagonal matrix 1 for v, 0 for x...)
	int* qmatbeg=new int[num_variables];
	int* qmatcnt=new int[num_variables];
	int* qmatind=new int[num_variables];
	double* qmatval=new double[num_variables];

	float64_t diag=2.0;

	for (int32_t i=0; i<num_variables; i++)
	{
		if (i<num_dim) //|| (!use_bias && i<num_dim)) //xi
		//if ((i<num_dim-1) || (!use_bias && i<num_dim)) //xi
		{
			qmatbeg[i]=i;
			qmatcnt[i]=1;
			qmatind[i]=i;
			qmatval[i]=diag;
		}
		else
		{
			//qmatbeg[i]= (use_bias) ? (num_dim-1) : (num_dim);
			qmatbeg[i]= num_dim;
			qmatcnt[i]=0;
			qmatind[i]=0;
			qmatval[i]=0;
		}
	}

	if (result)
		result = CPXcopyquad(env, lp, qmatbeg, qmatcnt, qmatind, qmatval) == 0;

	delete[] qmatbeg;
	delete[] qmatcnt;
	delete[] qmatind;
	delete[] qmatval;

	if (!result)
		SG_ERROR("CPXcopyquad failed.\n");

	//write_problem("problem.lp");
	//write_Q("problem.qp");

	return result;
}

bool CCplex::setup_lpboost(float64_t C, int32_t num_cols)
{
	init(E_LINEAR);
	int32_t status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, 1); //primal simplex
	if (status)
		SG_ERROR( "Failure to select dual lp optimization, error %d.\n", status);

	double* obj=new double[num_cols];
	double* lb=new double[num_cols];
	double* ub=new double[num_cols];

	for (int32_t i=0; i<num_cols; i++)
	{
		obj[i]=-1;
		lb[i]=0;
		ub[i]=C;
	}

	status = CPXnewcols(env, lp, num_cols, obj, lb, ub, NULL, NULL);
	if ( status )
	{
		char  errmsg[1024];
		CPXgeterrorstring (env, status, errmsg);
		SG_ERROR( "%s", errmsg);
	}
	delete[] obj;
	delete[] lb;
	delete[] ub;
	return status==0;
}

bool CCplex::add_lpboost_constraint(
	float64_t factor, SGSparseMatrixEntry<float64_t>* h, int32_t len, int32_t ulen,
	CLabels* label)
{
	int amatbeg[1]; /* for calling external lib */
	int amatind[len+1]; /* for calling external lib */
	double amatval[len+1]; /* for calling external lib */
	double rhs[1]; /* for calling external lib */
	char sense[1];

	amatbeg[0]=0;
	rhs[0]=1;
	sense[0]='L';

	for (int32_t i=0; i<len; i++)
	{
		int32_t idx=h[i].feat_index;
		float64_t val=factor*h[i].entry;
		amatind[i]=idx;
		amatval[i]=label->get_label(idx)*val;
	}

	int32_t status = CPXaddrows (env, lp, 0, 1, len, rhs, sense, amatbeg, amatind, amatval, NULL, NULL);

	if ( status ) 
		SG_ERROR( "Failed to add the new row.\n");

	return status == 0;
}

bool CCplex::setup_lpm(
	float64_t C, CSparseFeatures<float64_t>* x, CLabels* y, bool use_bias)
{
	ASSERT(x);
	ASSERT(y);
	int32_t num_vec=y->get_num_labels();
	int32_t num_feat=x->get_num_features();
	int64_t nnz=x->get_num_nonzero_entries();

	//number of variables: b,w+,w-,xi concatenated
	int32_t num_dims=1+2*num_feat+num_vec;
	int32_t num_constraints=num_vec; 

	float64_t* lb=new float64_t[num_dims];
	float64_t* ub=new float64_t[num_dims];
	float64_t* f=new float64_t[num_dims];
	float64_t* b=new float64_t[num_dims];

	//number of non zero entries in A (b,w+,w-,xi)
	int64_t amatsize=((int64_t) num_vec)+nnz+nnz+num_vec; 

	int* amatbeg=new int[num_dims]; /* for calling external lib */
	int* amatcnt=new int[num_dims]; /* for calling external lib */
	int* amatind=new int[amatsize]; /* for calling external lib */
	double* amatval= new double[amatsize]; /* for calling external lib */

	for (int32_t i=0; i<num_dims; i++)
	{
		if (i==0) //b
		{
			if (use_bias)
			{
				lb[i]=-CPX_INFBOUND;
				ub[i]=+CPX_INFBOUND;
			}
			else
			{
				lb[i]=0;
				ub[i]=0;
			}
			f[i]=0;
		}
		else if (i<2*num_feat+1) //w+,w-
		{
			lb[i]=0;
			ub[i]=CPX_INFBOUND;
			f[i]=1;
		}
		else //xi
		{
			lb[i]=0;
			ub[i]=CPX_INFBOUND;
			f[i]=C;
		}
	}

	for (int32_t i=0; i<num_constraints; i++)
		b[i]=-1;

	char* sense=new char[num_constraints];
	memset(sense,'L',sizeof(char)*num_constraints);

	//construct A
	int64_t offs=0;

	//b part of A
	amatbeg[0]=offs;
	amatcnt[0]=num_vec;

	for (int32_t i=0; i<num_vec; i++)
	{
		amatind[offs]=i;
		amatval[offs]=-y->get_label(i);
		offs++;
	}

	//w+ and w- part of A
	int32_t num_sfeat=0;
	int32_t num_svec=0;
	SGSparseMatrix<float64_t>* sfeat= x->get_transposed(num_sfeat, num_svec);
	ASSERT(sfeat);

	for (int32_t i=0; i<num_svec; i++)
	{
		amatbeg[i+1]=offs;
		amatcnt[i+1]=sfeat[i].num_feat_entries;

		for (int32_t j=0; j<sfeat[i].num_feat_entries; j++)
		{
			int32_t row=sfeat[i].features[j].feat_index;
			float64_t val=sfeat[i].features[j].entry;

			amatind[offs]=row;
			amatval[offs]=-y->get_label(row)*val;
			offs++;
		}
	}

	for (int32_t i=0; i<num_svec; i++)
	{
		amatbeg[num_svec+i+1]=offs;
		amatcnt[num_svec+i+1]=sfeat[i].num_feat_entries;

		for (int32_t j=0; j<sfeat[i].num_feat_entries; j++)
		{
			int32_t row=sfeat[i].features[j].feat_index;
			float64_t val=sfeat[i].features[j].entry;

			amatind[offs]=row;
			amatval[offs]=y->get_label(row)*val;
			offs++;
		}
	}

	x->clean_tsparse(sfeat, num_svec);

	//xi part of A
	for (int32_t i=0; i<num_vec; i++)
	{
		amatbeg[1+2*num_feat+i]=offs;
		amatcnt[1+2*num_feat+i]=1;
		amatind[offs]=i;
		amatval[offs]=-1;
		offs++;
	}

	int32_t status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, 1); //barrier
	if (status)
		SG_ERROR( "Failure to select barrier optimization, error %d.\n", status);
	CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_ON);

	bool result = CPXcopylp(env, lp, num_dims, num_constraints, CPX_MIN, 
			f, b, sense, amatbeg, amatcnt, amatind, amatval, lb, ub, NULL) == 0;


	delete[] amatval;
	delete[] amatcnt;
	delete[] amatind;
	delete[] amatbeg;
	delete[] b;
	delete[] f;
	delete[] ub;
	delete[] lb;

	return result;
}

bool CCplex::cleanup()
{
	bool result=false;

	if (lp)
	{
		int32_t status = CPXfreeprob(env, &lp);
		lp = NULL;
		lp_initialized = false;

		if (status)
			SG_WARNING( "CPXfreeprob failed, error code %d.\n", status);
		else
			result = true;
	}

	if (env)
	{
		int32_t status = CPXcloseCPLEX (&env);
		env=NULL;
		
		if (status)
		{
			char  errmsg[1024];
			SG_WARNING( "Could not close CPLEX environment.\n");
			CPXgeterrorstring (env, status, errmsg);
			SG_WARNING( "%s", errmsg);
		}
		else
			result = true;
	}
	return result;
}

bool CCplex::dense_to_cplex_sparse(
	float64_t* H, int32_t rows, int32_t cols, int* &qmatbeg, int* &qmatcnt,
	int* &qmatind, double* &qmatval)
{
	qmatbeg=new int[cols];
	qmatcnt=new int[cols];
	qmatind=new int[cols*rows];
	qmatval = H;

	if (!(qmatbeg && qmatcnt && qmatind))
	{
		delete[] qmatbeg;
		delete[] qmatcnt;
		delete[] qmatind;
		return false;
	}

	for (int32_t i=0; i<cols; i++)
	{
		qmatcnt[i]=rows;
		qmatbeg[i]=i*rows;
		for (int32_t j=0; j<rows; j++)
			qmatind[i*rows+j]=j;
	}

	return true;
}

bool CCplex::setup_lp(
	float64_t* objective, float64_t* constraints_mat, int32_t rows,
	int32_t cols, float64_t* rhs, float64_t* lb, float64_t* ub)
{
	bool result=false;

	int* qmatbeg=NULL;
	int* qmatcnt=NULL;
	int* qmatind=NULL;
	double* qmatval=NULL;

	char* sense = NULL;

	if (constraints_mat==0)
	{
		ASSERT(rows==0);
		rows=1;
		float64_t dummy=0;
		rhs=&dummy;
		sense=new char[rows];
		memset(sense,'L',sizeof(char)*rows);
		constraints_mat=new float64_t[cols];
		memset(constraints_mat, 0, sizeof(float64_t)*cols);

		result=dense_to_cplex_sparse(constraints_mat, 0, cols, qmatbeg, qmatcnt, qmatind, qmatval);
		ASSERT(result);
		result = CPXcopylp(env, lp, cols, rows, CPX_MIN, 
				objective, rhs, sense, qmatbeg, qmatcnt, qmatind, qmatval, lb, ub, NULL) == 0;
		delete[] constraints_mat;
	}
	else
	{
		sense=new char[rows];
		memset(sense,'L',sizeof(char)*rows);
		result=dense_to_cplex_sparse(constraints_mat, rows, cols, qmatbeg, qmatcnt, qmatind, qmatval);
		result = CPXcopylp(env, lp, cols, rows, CPX_MIN, 
				objective, rhs, sense, qmatbeg, qmatcnt, qmatind, qmatval, lb, ub, NULL) == 0;
	}

	delete[] sense;
	delete[] qmatbeg;
	delete[] qmatcnt;
	delete[] qmatind;

	if (!result)
		SG_WARNING("CPXcopylp failed.\n");

	return result;
}

bool CCplex::setup_qp(float64_t* H, int32_t dim)
{
	int* qmatbeg=NULL;
	int* qmatcnt=NULL;
	int* qmatind=NULL;
	double* qmatval=NULL;
	bool result=dense_to_cplex_sparse(H, dim, dim, qmatbeg, qmatcnt, qmatind, qmatval);
	if (result)
		result = CPXcopyquad(env, lp, qmatbeg, qmatcnt, qmatind, qmatval) == 0;

	delete[] qmatbeg;
	delete[] qmatcnt;
	delete[] qmatind;

	if (!result)
		SG_WARNING("CPXcopyquad failed.\n");

	return result;
}

bool CCplex::optimize(float64_t* sol, float64_t* lambda)
{
	int      solnstat; /* for calling external lib */
	double   objval;
	int status=1; /* for calling external lib */

	if (problem_type==E_QP)
		status = CPXqpopt (env, lp);
	else if (problem_type == E_LINEAR)
		status = CPXlpopt (env, lp);

	if (status)
		SG_WARNING( "Failed to optimize QP.\n");

	status = CPXsolution (env, lp, &solnstat, &objval, sol, lambda, NULL, NULL);

	//SG_PRINT("obj:%f\n", objval);

	return (status==0);
}
#endif

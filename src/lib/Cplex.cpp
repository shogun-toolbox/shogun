/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef USE_CPLEX
#include <unistd.h>

#include "lib/Cplex.h"
#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Mathematics.h"

CCplex::CCplex() : CSGObject(), env(NULL), lp(NULL), lp_initialized(false)
{
}

CCplex::~CCplex()
{
	cleanup();
}

bool CCplex::init(E_PROB_TYPE typ, INT timeout)
{
	problem_type=typ;

	while (env==NULL)
	{
		int status = 1;
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

			status = CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_ON);
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

					//status = CPXsetdblparam (env, CPX_PARAM_TILIM, 0.5);
					//if (status)
					//	SG_ERROR( "Failure to set time limit %d.\n", status);
				}
			}
		}
	}

	return (lp != NULL) && (env != NULL);
}

bool CCplex::setup_subgradientlpm_QP(DREAL C, CLabels* labels, CSparseFeatures<DREAL>* features, INT* idx_bound, INT num_bound,
		INT* w_zero, INT num_zero,
		DREAL* vee, INT num_dim,
		bool use_bias)
{
	const int cmatsize=10*1024*1024; //no more than 10mil. elements
	bool result=false;
	INT num_variables=num_dim + num_bound+num_zero; // xi, beta

	// setup LP part
	DREAL* lb=new DREAL[num_variables];
	ASSERT(lb);
	DREAL* ub=new DREAL[num_variables];
	ASSERT(ub);
	DREAL* obj=new DREAL[num_variables];
	ASSERT(obj);

	char* sense = new char[num_dim];
	ASSERT(sense);

	int* cmatbeg=new int[num_variables];
	ASSERT(cmatbeg);
	int* cmatcnt=new int[num_variables];
	ASSERT(cmatcnt);
	int* cmatind=new int[cmatsize];
	ASSERT(cmatind);
	double* cmatval=new double[cmatsize];
	ASSERT(cmatval);

	for (INT i=0; i<num_variables; i++)
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

	INT offs=0;
	for (INT i=0; i<num_variables; i++)
	{
		if (i<num_dim) //sum_xi
		{
			sense[i]='E';
			cmatbeg[i]=offs;
			cmatcnt[i]=1;
			cmatind[offs]=offs;
			cmatval[offs]=1.0;
			offs++;
		}
		else if (i<num_dim+num_zero) // Z_w*beta_w
		{
			cmatbeg[i]=offs;
			cmatcnt[i]=1;
			cmatind[offs]=w_zero[i-num_dim];
			cmatval[offs]=1.0;
			offs++;
		}
		else // Z_x*beta_x
		{
			INT idx=i-num_dim-num_zero;
			INT vlen=0;
			bool vfree=false;
			TSparseEntry<DREAL>* vec=features->get_sparse_feature_vector(idx, vlen, vfree);

			cmatbeg[i]=offs;
			cmatcnt[i]=vlen;

			DREAL val= -C*labels->get_label(idx);

			for (INT j=0; j<vlen; j++)
			{
				cmatind[offs]=vec[j].feat_index;
				cmatval[offs]=val*vec[j].entry;
				offs++;
			}

			features->free_feature_vector(vec, idx, vfree);
		}
	}

	result = CPXcopylp(env, lp, num_variables, 0, CPX_MIN, 
			obj, vee, sense, cmatbeg, cmatcnt, cmatind, cmatval, lb, ub, NULL) == 0;

	if (!result)
		SG_ERROR("CPXcopylp failed.\n");

	write_problem("problem.lp");

	delete[] sense;
	delete[] lb;
	delete[] ub;
	delete[] obj;
	delete[] cmatbeg;
	delete[] cmatcnt;
	delete[] cmatind;
	delete[] cmatval;

	// setup QP part (diagonal matrix 1 for v, 0 for x...)
	int* qmatbeg=new int[num_variables];
	ASSERT(qmatbeg);
	int* qmatcnt=new int[num_variables];
	ASSERT(qmatcnt);
	int* qmatind=new int[num_variables];
	ASSERT(qmatind);
	double* qmatval=new double[num_variables];
	ASSERT(qmatval);

	for (INT i=0; i<num_variables; i++)
	{
		if ((i<num_dim-1) || (!use_bias && i<num_dim)) //xi
		{
			qmatbeg[i]=i;
			qmatcnt[i]=1;
			qmatind[i]=i;
			qmatval[i]=1.0;
		}
		else
		{
			qmatbeg[i]= (use_bias) ? (num_dim-1) : (num_dim);
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

	return result;
}

bool CCplex::setup_lpboost(DREAL C, INT num_cols)
{
	init(E_LINEAR);
	INT status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, 1); //primal simplex
	if (status)
		SG_ERROR( "Failure to select dual lp optimization, error %d.\n", status);

	double obj[num_cols];
	double lb[num_cols];
	double ub[num_cols];

	for (INT i=0; i<num_cols; i++)
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
	return status==0;
}

bool CCplex::add_lpboost_constraint(DREAL factor, TSparseEntry<DREAL>* h, INT len, INT ulen, CLabels* label)
{
	int amatbeg[1];
	int amatind[len+1];
	double amatval[len+1];
	double rhs[1];
	char sense[1];

	amatbeg[0]=0;
	rhs[0]=1;
	sense[0]='L';

	for (INT i=0; i<len; i++)
	{
		INT idx=h[i].feat_index;
		DREAL val=factor*h[i].entry;
		amatind[i]=idx;
		amatval[i]=label->get_label(idx)*val;
	}

	INT status = CPXaddrows (env, lp, 0, 1, len, rhs, sense, amatbeg, amatind, amatval, NULL, NULL);

	if ( status ) 
		SG_ERROR( "Failed to add the new row.\n");

	return status == 0;
}

bool CCplex::setup_lpm(DREAL C, CSparseFeatures<DREAL>* x, CLabels* y, bool use_bias)
{
	ASSERT(x);
	ASSERT(y);
	INT num_vec=y->get_num_labels();
	INT num_feat=x->get_num_features();
	LONG nnz=x->get_num_nonzero_entries();

	//number of variables: b,w+,w-,xi concatenated
	INT num_dims=1+2*num_feat+num_vec;
	INT num_constraints=num_vec; 

	DREAL* lb=new DREAL[num_dims];
	ASSERT(lb);
	DREAL* ub=new DREAL[num_dims];
	ASSERT(ub);
	DREAL* f=new DREAL[num_dims];
	ASSERT(f);
	DREAL* b=new DREAL[num_dims];
	ASSERT(b);

	//number of non zero entries in A (b,w+,w-,xi)
	LONG amatsize=((LONG) num_vec)+nnz+nnz+num_vec; 

	int* amatbeg=new int[num_dims];
	ASSERT(amatbeg);
	int* amatcnt=new int[num_dims];
	ASSERT(amatcnt);
	int* amatind=new int[amatsize];
	ASSERT(amatind);
	double* amatval= new double[amatsize];
	ASSERT(amatval);

	for (INT i=0; i<num_dims; i++)
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

	for (INT i=0; i<num_constraints; i++)
		b[i]=-1;

	char* sense = new char[num_constraints];
	ASSERT(sense);
	memset(sense,'L',sizeof(char)*num_constraints);

	//construct A
	LONG offs=0;

	//b part of A
	amatbeg[0]=offs;
	amatcnt[0]=num_vec;

	for (INT i=0; i<num_vec; i++)
	{
		amatind[offs]=i;
		amatval[offs]=-y->get_label(i);
		offs++;
	}

	//w+ and w- part of A
	INT num_sfeat=0;
	INT num_svec=0;
	TSparse<DREAL>* sfeat= x->get_transposed(num_sfeat, num_svec);
	ASSERT(sfeat);

	for (INT i=0; i<num_svec; i++)
	{
		amatbeg[i+1]=offs;
		amatcnt[i+1]=sfeat[i].num_feat_entries;

		for (INT j=0; j<sfeat[i].num_feat_entries; j++)
		{
			INT row=sfeat[i].features[j].feat_index;
			DREAL val=sfeat[i].features[j].entry;

			amatind[offs]=row;
			amatval[offs]=-y->get_label(row)*val;
			offs++;
		}
	}

	for (INT i=0; i<num_svec; i++)
	{
		amatbeg[num_svec+i+1]=offs;
		amatcnt[num_svec+i+1]=sfeat[i].num_feat_entries;

		for (INT j=0; j<sfeat[i].num_feat_entries; j++)
		{
			INT row=sfeat[i].features[j].feat_index;
			DREAL val=sfeat[i].features[j].entry;

			amatind[offs]=row;
			amatval[offs]=y->get_label(row)*val;
			offs++;
		}
	}

	x->clean_tsparse(sfeat, num_svec);

	//xi part of A
	for (INT i=0; i<num_vec; i++)
	{
		amatbeg[1+2*num_feat+i]=offs;
		amatcnt[1+2*num_feat+i]=1;
		amatind[offs]=i;
		amatval[offs]=-1;
		offs++;
	}

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
		INT status = CPXfreeprob(env, &lp);
		lp = NULL;
		lp_initialized = false;

		if (status)
			SG_WARNING( "CPXfreeprob failed, error code %d.\n", status);
		else
			result = true;
	}

	if (env)
	{
		INT status = CPXcloseCPLEX (&env);
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

bool CCplex::dense_to_cplex_sparse(DREAL* H, INT rows, INT cols, int* &qmatbeg, int* &qmatcnt, int* &qmatind, double* &qmatval)
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

	for (INT i=0; i<cols; i++)
	{
		qmatcnt[i]=rows;
		qmatbeg[i]=i*rows;
		for (INT j=0; j<rows; j++)
			qmatind[i*rows+j]=j;
	}

	return true;
}

bool CCplex::setup_lp(DREAL* objective, DREAL* constraints_mat, INT rows, INT cols, DREAL* rhs, DREAL* lb, DREAL* ub)
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
		DREAL dummy=0;
		rhs=&dummy;
		sense=new char[rows];
		ASSERT(sense);
		memset(sense,'L',sizeof(char)*rows);
		constraints_mat=new DREAL[cols];
		ASSERT(constraints_mat);
		memset(constraints_mat, 0, sizeof(DREAL)*cols);

		result=dense_to_cplex_sparse(constraints_mat, 0, cols, qmatbeg, qmatcnt, qmatind, qmatval);
		ASSERT(result);
		result = CPXcopylp(env, lp, cols, rows, CPX_MIN, 
				objective, rhs, sense, qmatbeg, qmatcnt, qmatind, qmatval, lb, ub, NULL) == 0;
		delete[] constraints_mat;
	}
	else
	{
		sense=new char[rows];
		ASSERT(sense);
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

bool CCplex::setup_qp(DREAL* H, INT dim)
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

bool CCplex::optimize(DREAL* sol, DREAL* lambda)
{
	int      solnstat;
	double   objval;
	int status=1;

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

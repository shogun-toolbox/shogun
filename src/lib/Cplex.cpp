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

			status = CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_OFF);
			if (status)
				SG_ERROR( "Failure to turn off screen indicator, error %d.\n", status);

			{
				status = CPXsetintparam (env, CPX_PARAM_DATACHECK, CPX_OFF);
				if (status)
					SG_ERROR( "Failure to turn on data checking, error %d.\n", status);
				else
				{
					lp = CPXcreateprob (env, &status, "light");

					if ( lp == NULL )
						SG_ERROR( "Failed to create optimization problem.\n");
					else
						CPXchgobjsen (env, lp, CPX_MIN);  /* Problem is minimization */
				}
			}
		}
	}

	return (lp != NULL) && (env != NULL);
}

bool CCplex::setup_lpm(DREAL C, CSparseFeatures<DREAL>* x, CLabels* y)
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
			lb[i]=-CMath::ALMOST_INFTY;
			ub[i]=+CMath::ALMOST_INFTY;
			f[i]=0;
		}
		else if (i<2*num_feat+1) //w+,w-
		{
			lb[i]=0;
			ub[i]=CMath::ALMOST_INFTY;
			f[i]=1;
		}
		else //xi
		{
			lb[i]=0;
			ub[i]=CMath::ALMOST_INFTY;
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

bool CCplex::optimize(DREAL* sol, INT dim)
{
	int      solnstat;//, solnmethod, solntype;
	double   objval;
	int status=1;

	//status = CPXsetdblparam (env, CPX_PARAM_TILIM, 0.5);
	//if (status)
	//	SG_ERROR( "Failure to set time limit %d.\n", status);

	if (problem_type==QP)
		status = CPXsetintparam (env, CPX_PARAM_QPMETHOD, 0);
	else if (problem_type == LINEAR)
		status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, 0);

	if (status)
		SG_ERROR( "Failure to select dual lp/qp optimization, error %d.\n", status);

	if (problem_type==QP)
		status = CPXqpopt (env, lp);
	else if (problem_type == LINEAR)
		status = CPXlpopt (env, lp);


	if (status)
		SG_WARNING( "Failed to optimize QP.\n");

	status = CPXsolution (env, lp, &solnstat, &objval, sol, NULL, NULL, NULL);

//	if ( status )
//		SG_ERROR("CPXsolution failed.\n");
//
//	solnstat = CPXgetstat (env, lp);
//
//	if ( solnstat == CPX_STAT_UNBOUNDED )
//		SG_INFO( "Model is unbounded\n");
//	else if ( solnstat == CPX_STAT_INFEASIBLE )
//		SG_INFO( "Model is infeasible\n");
//	else if ( solnstat == CPX_STAT_INForUNBD )
//		SG_INFO( "Model is infeasible or unbounded\n");
//
//	status = CPXsolninfo (env, lp, &solnmethod, &solntype, NULL, NULL);
//	if ( status )
//		SG_ERROR( "Failed to obtain solution info.\n");

	return (status==0);
}
#endif

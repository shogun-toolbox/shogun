/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/Signal.h"
#include "classifier/svm/MKL.h"
#include "classifier/svm/LibSVM.h"

CMKL::CMKL(CSVM* s)
	: CSVM(), svm(NULL), C_mkl(0), mkl_norm(1), mkl_iterations(0), mkl_epsilon(1e-5), interleaved_optimization(false)
{
	set_constraint_generator(s);
#ifdef USE_CPLEX
	lp_cplex = NULL ;
	env = NULL ;
#endif

#ifdef USE_GLPK
	lp_glpk = NULL;
#endif
	svm=s;

	if (!svm)
		svm=new CLibSVM();
	SG_REF(svm);

	lp_initialized = false ;
}

CMKL::~CMKL()
{
	svm->set_callback_function(NULL);
	SG_UNREF(svm);
}

void CMKL::init_solver()
{
#ifdef USE_CPLEX
	cleanup_cplex();

	if (get_solver_type()==ST_CPLEX || get_solver_type()==ST_AUTO)
		init_cplex();
#endif

#ifdef USE_GLPK
	cleanup_glpk();

	if (get_solver_type() == ST_GLPK ||
				( mkl_norm == 1 && get_solver_type()==ST_AUTO))
		init_glpk();
#endif
}

#ifdef USE_CPLEX
bool CMKL::init_cplex()
{
	while (env==NULL)
	{
		SG_INFO( "trying to initialize CPLEX\n") ;

		int status = 0; // calling external lib
		env = CPXopenCPLEX (&status);

		if ( env == NULL )
		{
			char  errmsg[1024];
			SG_WARNING( "Could not open CPLEX environment.\n");
			CPXgeterrorstring (env, status, errmsg);
			SG_WARNING( "%s", errmsg);
			SG_WARNING( "retrying in 60 seconds\n");
			sleep(60);
		}
		else
		{
			// select dual simplex based optimization
			status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, CPX_ALG_DUAL);
			if ( status )
			{
            SG_ERROR( "Failure to select dual lp optimization, error %d.\n", status);
			}
			else
			{
				status = CPXsetintparam (env, CPX_PARAM_DATACHECK, CPX_ON);
				if ( status )
				{
					SG_ERROR( "Failure to turn on data checking, error %d.\n", status);
				}	
				else
				{
					lp_cplex = CPXcreateprob (env, &status, "light");

					if ( lp_cplex == NULL )
						SG_ERROR( "Failed to create LP.\n");
					else
						CPXchgobjsen (env, lp_cplex, CPX_MIN);  /* Problem is minimization */
				}
			}
		}
	}

	return (lp_cplex != NULL) && (env != NULL);
}

bool CMKL::cleanup_cplex()
{
	bool result=false;

	if (lp_cplex)
	{
		int32_t status = CPXfreeprob(env, &lp_cplex);
		lp_cplex = NULL;
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
#endif

#ifdef USE_GLPK
bool CMKL::init_glpk()
{
	lp_glpk = lpx_create_prob();
	lpx_set_obj_dir(lp_glpk, LPX_MIN);
	lpx_set_int_parm(lp_glpk, LPX_K_DUAL, GLP_ON );
	lpx_set_int_parm(lp_glpk, LPX_K_PRESOL, GLP_ON );

	glp_term_out(GLP_OFF);
	return (lp_glpk != NULL);
}

bool CMKL::cleanup_glpk()
{
	lp_initialized = false;
	if (lp_glpk)
		lpx_delete_prob(lp_glpk);
	lp_glpk = NULL;
	return true;
}

bool CMKL::check_lpx_status(LPX *lp)
{
	int status = lpx_get_status(lp);

	if (status==LPX_INFEAS)
	{
		SG_PRINT("solution is infeasible!\n");
		return false;
	}
	else if(status==LPX_NOFEAS)
	{
		SG_PRINT("problem has no feasible solution!\n");
		return false;
	}
	return true;
}
#endif // USE_GLPK

bool CMKL::train()
{
	ASSERT(labels && labels->get_num_labels());
	ASSERT(labels->is_two_class_labeling());

	int32_t num_label=labels->get_num_labels();

	SG_INFO("%d trainlabels\n", num_label);
	if (mkl_epsilon<=0)
		mkl_epsilon=1e-2 ;

	SG_INFO("mkl_epsilon = %1.1e\n", mkl_epsilon) ;
	SG_INFO("C_mkl = %1.1e\n", C_mkl) ;
	SG_INFO("mkl_norm = %1.3e\n", mkl_norm);

#ifdef USE_CPLEX
	cleanup_cplex();

	if (get_solver_type()==ST_CPLEX || get_solver_type()==ST_AUTO)
		init_cplex();
#endif

#ifdef USE_GLPK
	if (get_solver_type()==ST_GLPK || get_solver_type()==ST_AUTO)
		init_glpk();
#endif

	mkl_iterations = 0;
	
	if (interleaved_optimization)
	{
		if (svm->get_classifier_type() != CT_LIGHT)
		{
			SG_ERROR("Interleaved MKL optimization is currently "
					"only supported with SVMlight\n");
		}
		svm->set_callback_function(perform_mkl_step);
		svm->train();
	}
	else
	{
		int32_t num_kernels = kernel->get_num_subkernels();
		float64_t* sumw = new float64_t[num_kernels];

		svm->set_bias_enabled(get_bias_enabled());
		svm->set_epsilon(get_epsilon());
		svm->set_max_train_time(get_max_train_time());
		svm->set_nu(get_nu());
		svm->set_C(get_C1(), get_C2());
		svm->set_qpsize(get_qpsize());
		svm->set_shrinking_enabled(get_shrinking_enabled());
		svm->set_linadd_enabled(get_linadd_enabled());
		svm->set_batch_computation_enabled(get_batch_computation_enabled());
		svm->set_labels(labels);
		svm->set_kernel(kernel);

		while (true)
		{
			svm->train();

			float64_t suma=compute_sum_alpha();
			compute_sum_beta(sumw);

			mkl_iterations++;
			if (perform_mkl_step(sumw, suma) ||
					CSignal::cancel_computations())
				break;
		}

		delete[] sumw;
	}
#ifdef USE_CPLEX
	cleanup_cplex();
#endif
#ifdef USE_GLPK
	cleanup_glpk();
#endif

	int32_t nsv=svm->get_num_support_vectors();
	create_new_model(nsv);

	set_bias(svm->get_bias());
	for (int32_t i=0; i<nsv; i++)
	{
		set_alpha(i, svm->get_alpha(i));
		set_support_vector(i, svm->get_support_vector(i));
	}
	return true;
}

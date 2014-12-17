/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2004-2009 Soeren Sonnenburg, Gunnar Raetsch
 *                       Alexander Zien, Marius Kloft, Chen Guohua
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Ryota Tomioka (University of Tokyo)
 */

#include <list>
#include <shogun/lib/Signal.h>
#include <shogun/classifier/mkl/MKL.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/kernel/CombinedKernel.h>


using namespace shogun;

CMKL::CMKL(CSVM* s) : CSVM(), svm(NULL), C_mkl(0), mkl_norm(1), ent_lambda(0),
		mkl_block_norm(1),beta_local(NULL), mkl_iterations(0), mkl_epsilon(1e-5),
		interleaved_optimization(true), w_gap(1.0), rho(0)
{
	set_constraint_generator(s);
#ifdef USE_CPLEX
	lp_cplex = NULL ;
	env = NULL ;
#endif

#ifdef USE_GLPK
	lp_glpk = NULL;
	lp_glpk_parm = NULL;
#endif

	SG_DEBUG("creating MKL object %p\n", this)
	lp_initialized = false ;
}

CMKL::~CMKL()
{
	// -- Delete beta_local for ElasticnetMKL
	SG_FREE(beta_local);

	SG_DEBUG("deleting MKL object %p\n", this)
	if (svm)
		svm->set_callback_function(NULL, NULL);
	SG_UNREF(svm);
}

void CMKL::init_solver()
{
#ifdef USE_CPLEX
	cleanup_cplex();

	if (get_solver_type()==ST_CPLEX)
		init_cplex();
#endif

#ifdef USE_GLPK
	cleanup_glpk();

	if (get_solver_type() == ST_GLPK)
		init_glpk();
#endif
}

#ifdef USE_CPLEX
bool CMKL::init_cplex()
{
	while (env==NULL)
	{
		SG_INFO("trying to initialize CPLEX\n")

		int status = 0; // calling external lib
		env = CPXopenCPLEX (&status);

		if ( env == NULL )
		{
			char  errmsg[1024];
			SG_WARNING("Could not open CPLEX environment.\n")
			CPXgeterrorstring (env, status, errmsg);
			SG_WARNING("%s", errmsg)
			SG_WARNING("retrying in 60 seconds\n")
			sleep(60);
		}
		else
		{
			// select dual simplex based optimization
			status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, CPX_ALG_DUAL);
			if ( status )
			{
            SG_ERROR("Failure to select dual lp optimization, error %d.\n", status)
			}
			else
			{
				status = CPXsetintparam (env, CPX_PARAM_DATACHECK, CPX_ON);
				if ( status )
				{
					SG_ERROR("Failure to turn on data checking, error %d.\n", status)
				}
				else
				{
					lp_cplex = CPXcreateprob (env, &status, "light");

					if ( lp_cplex == NULL )
						SG_ERROR("Failed to create LP.\n")
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
			SG_WARNING("CPXfreeprob failed, error code %d.\n", status)
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
			SG_WARNING("Could not close CPLEX environment.\n")
			CPXgeterrorstring (env, status, errmsg);
			SG_WARNING("%s", errmsg)
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
	lp_glpk = glp_create_prob();
	glp_set_obj_dir(lp_glpk, GLP_MIN);

	lp_glpk_parm = SG_MALLOC(glp_smcp, 1);
	glp_init_smcp(lp_glpk_parm);
	lp_glpk_parm->meth = GLP_DUAL;
	lp_glpk_parm->presolve = GLP_ON;

	glp_term_out(GLP_OFF);
	return (lp_glpk != NULL);
}

bool CMKL::cleanup_glpk()
{
	lp_initialized = false;
	if (lp_glpk)
		glp_delete_prob(lp_glpk);
	lp_glpk = NULL;
	SG_FREE(lp_glpk_parm);
	return true;
}

bool CMKL::check_glp_status(glp_prob *lp)
{
	int status = glp_get_status(lp);

	if (status==GLP_INFEAS)
	{
		SG_PRINT("solution is infeasible!\n")
		return false;
	}
	else if(status==GLP_NOFEAS)
	{
		SG_PRINT("problem has no feasible solution!\n")
		return false;
	}
	return true;
}
#endif // USE_GLPK

bool CMKL::train_machine(CFeatures* data)
{
	ASSERT(kernel)
	ASSERT(m_labels && m_labels->get_num_labels())

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
		{
			SG_ERROR("%s::train_machine(): Number of training vectors (%d) does"
					" not match number of labels (%d)\n", get_name(),
					data->get_num_vectors(), m_labels->get_num_labels());
		}
		kernel->init(data, data);
	}

	init_training();
	if (!svm)
		SG_ERROR("No constraint generator (SVM) set\n")

	int32_t num_label=0;
	if (m_labels)
		num_label = m_labels->get_num_labels();

	SG_INFO("%d trainlabels (%ld)\n", num_label, m_labels)
	if (mkl_epsilon<=0)
		mkl_epsilon=1e-2 ;

	SG_INFO("mkl_epsilon = %1.1e\n", mkl_epsilon)
	SG_INFO("C_mkl = %1.1e\n", C_mkl)
	SG_INFO("mkl_norm = %1.3e\n", mkl_norm)
	SG_INFO("solver = %d\n", get_solver_type())
	SG_INFO("ent_lambda = %f\n", ent_lambda)
	SG_INFO("mkl_block_norm = %f\n", mkl_block_norm)

	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels();
	SG_INFO("num_kernels = %d\n", num_kernels)
	const float64_t* beta_const   = kernel->get_subkernel_weights(num_weights);
	float64_t* beta = SGVector<float64_t>::clone_vector(beta_const, num_weights);
	ASSERT(num_weights==num_kernels)

	if (get_solver_type()==ST_BLOCK_NORM &&
			mkl_block_norm>=1 &&
			mkl_block_norm<=2)
	{
		mkl_norm=mkl_block_norm/(2-mkl_block_norm);
		SG_WARNING("Switching to ST_DIRECT method with mkl_norm=%g\n", mkl_norm)
		set_solver_type(ST_DIRECT);
	}

	if (get_solver_type()==ST_ELASTICNET)
	{
	  // -- Initialize subkernel weights for Elasticnet MKL
	  SGVector<float64_t>::scale_vector(1/SGVector<float64_t>::qnorm(beta, num_kernels, 1.0), beta, num_kernels);

	  SG_FREE(beta_local);
	  beta_local = SGVector<float64_t>::clone_vector(beta, num_kernels);

	  elasticnet_transform(beta, ent_lambda, num_kernels);
	}
	else
	{
		SGVector<float64_t>::scale_vector(1/SGVector<float64_t>::qnorm(beta, num_kernels, mkl_norm),
				beta, num_kernels); //q-norm = 1
	}

	kernel->set_subkernel_weights(SGVector<float64_t>(beta, num_kernels, false));
	SG_FREE(beta);

	svm->set_bias_enabled(get_bias_enabled());
	svm->set_epsilon(get_epsilon());
	svm->set_max_train_time(get_max_train_time());
	svm->set_nu(get_nu());
	svm->set_C(get_C1(), get_C2());
	svm->set_qpsize(get_qpsize());
	svm->set_shrinking_enabled(get_shrinking_enabled());
	svm->set_linadd_enabled(get_linadd_enabled());
	svm->set_batch_computation_enabled(get_batch_computation_enabled());
	svm->set_labels(m_labels);
	svm->set_kernel(kernel);

#ifdef USE_CPLEX
	cleanup_cplex();

	if (get_solver_type()==ST_CPLEX)
		init_cplex();
#endif

#ifdef USE_GLPK
	if (get_solver_type()==ST_GLPK)
		init_glpk();
#endif

	mkl_iterations = 0;
	CSignal::clear_cancel();

	training_time_clock.start();

	if (interleaved_optimization)
	{
		if (svm->get_classifier_type() != CT_LIGHT && svm->get_classifier_type() != CT_SVRLIGHT)
		{
			SG_ERROR("Interleaved MKL optimization is currently "
					"only supported with SVMlight\n");
		}

		//Note that there is currently no safe way to properly resolve this
		//situation: the mkl object hands itself as a reference to the svm which
		//in turn increases the reference count of mkl and when done decreases
		//the count. Thus we have to protect the mkl object from deletion by
		//ref()'ing it before we set the callback function and should also
		//unref() it afterwards. However, when the reference count was zero
		//before this unref() might actually try to destroy this (crash ahead)
		//but if we don't actually unref() the object we might leak memory...
		//So as a workaround we only unref when the reference count was >1
		//before.
#ifdef USE_REFERENCE_COUNTING
		int32_t refs=this->ref();
#endif
		svm->set_callback_function(this, perform_mkl_step_helper);
		svm->train();
		SG_DONE()
		svm->set_callback_function(NULL, NULL);
#ifdef USE_REFERENCE_COUNTING
		if (refs>1)
			this->unref();
#endif
	}
	else
	{
		float64_t* sumw = SG_MALLOC(float64_t, num_kernels);



		while (true)
		{
			svm->train();

			float64_t suma=compute_sum_alpha();
			compute_sum_beta(sumw);

			if((training_time_clock.cur_time_diff()>get_max_train_time ())&&(get_max_train_time ()>0))
			{
				SG_SWARNING("MKL Algorithm terminates PREMATURELY due to current training time exceeding get_max_train_time ()= %f . It may have not converged yet!\n",get_max_train_time ())
				break;
			}


			mkl_iterations++;
			if (perform_mkl_step(sumw, suma) || CSignal::cancel_computations())
				break;
		}

		SG_FREE(sumw);
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


void CMKL::set_mkl_norm(float64_t norm)
{

  if (norm<1)
    SG_ERROR("Norm must be >= 1, e.g., 1-norm is the standard MKL; norms>1 nonsparse MKL\n")

  mkl_norm = norm;
}

void CMKL::set_elasticnet_lambda(float64_t lambda)
{
  if (lambda>1 || lambda<0)
    SG_ERROR("0<=lambda<=1\n")

  if (lambda==0)
    lambda = 1e-6;
  else if (lambda==1.0)
    lambda = 1.0-1e-6;

  ent_lambda=lambda;
}

void CMKL::set_mkl_block_norm(float64_t q)
{
  if (q<1)
    SG_ERROR("1<=q<=inf\n")

  mkl_block_norm=q;
}

bool CMKL::perform_mkl_step(
		const float64_t* sumw, float64_t suma)
{
	if((training_time_clock.cur_time_diff()>get_max_train_time ())&&(get_max_train_time ()>0))
	{
		SG_SWARNING("MKL Algorithm terminates PREMATURELY due to current training time exceeding get_max_train_time ()= %f . It may have not converged yet!\n",get_max_train_time ())
		return true;
	}

	int32_t num_kernels = kernel->get_num_subkernels();
	int32_t nweights=0;
	const float64_t* old_beta = kernel->get_subkernel_weights(nweights);
	ASSERT(nweights==num_kernels)
	float64_t* beta = SG_MALLOC(float64_t, num_kernels);

#if defined(USE_CPLEX) || defined(USE_GLPK)
	int32_t inner_iters=0;
#endif
	float64_t mkl_objective=0;

	mkl_objective=-suma;
	for (int32_t i=0; i<num_kernels; i++)
	{
		beta[i]=old_beta[i];
		mkl_objective+=old_beta[i]*sumw[i];
	}

	w_gap = CMath::abs(1-rho/mkl_objective) ;

	if( (w_gap >= mkl_epsilon) ||
	    (get_solver_type()==ST_AUTO || get_solver_type()==ST_NEWTON || get_solver_type()==ST_DIRECT ) || get_solver_type()==ST_ELASTICNET || get_solver_type()==ST_BLOCK_NORM)
	{
		if (get_solver_type()==ST_AUTO || get_solver_type()==ST_DIRECT)
		{
			rho=compute_optimal_betas_directly(beta, old_beta, num_kernels, sumw, suma, mkl_objective);
		}
		else if (get_solver_type()==ST_BLOCK_NORM)
		{
			rho=compute_optimal_betas_block_norm(beta, old_beta, num_kernels, sumw, suma, mkl_objective);
		}
		else if (get_solver_type()==ST_ELASTICNET)
		{
			// -- Direct update of subkernel weights for ElasticnetMKL
			// Note that ElasticnetMKL solves a different optimization
			// problem from the rest of the solver types
			rho=compute_optimal_betas_elasticnet(beta, old_beta, num_kernels, sumw, suma, mkl_objective);
		}
		else if (get_solver_type()==ST_NEWTON)
			rho=compute_optimal_betas_newton(beta, old_beta, num_kernels, sumw, suma, mkl_objective);
#ifdef USE_CPLEX
		else if (get_solver_type()==ST_CPLEX)
			rho=compute_optimal_betas_via_cplex(beta, old_beta, num_kernels, sumw, suma, inner_iters);
#endif
#ifdef USE_GLPK
		else if (get_solver_type()==ST_GLPK)
			rho=compute_optimal_betas_via_glpk(beta, old_beta, num_kernels, sumw, suma, inner_iters);
#endif
		else
			SG_ERROR("Solver type not supported (not compiled in?)\n")

		w_gap = CMath::abs(1-rho/mkl_objective) ;
	}

	kernel->set_subkernel_weights(SGVector<float64_t>(beta, num_kernels, false));
	SG_FREE(beta);

	return converged();
}

float64_t CMKL::compute_optimal_betas_elasticnet(
  float64_t* beta, const float64_t* old_beta, const int32_t num_kernels,
  const float64_t* sumw, const float64_t suma,
  const float64_t mkl_objective )
{
	const float64_t epsRegul = 0.01;  // fraction of root mean squared deviation
	float64_t obj;
	float64_t Z;
	float64_t preR;
	int32_t p;
	int32_t nofKernelsGood;

	// --- optimal beta
	nofKernelsGood = num_kernels;

	Z = 0;
	for (p=0; p<num_kernels; ++p )
	{
		if (sumw[p] >= 0.0 && old_beta[p] >= 0.0 )
		{
			beta[p] = CMath::sqrt(sumw[p]*old_beta[p]*old_beta[p]);
			Z += beta[p];
		}
		else
		{
			beta[p] = 0.0;
			--nofKernelsGood;
		}
		ASSERT( beta[p] >= 0 )
	}

	// --- normalize
	SGVector<float64_t>::scale_vector(1.0/Z, beta, num_kernels);

	// --- regularize & renormalize

	preR = 0.0;
	for( p=0; p<num_kernels; ++p )
		preR += CMath::pow( beta_local[p] - beta[p], 2.0 );
	const float64_t R = CMath::sqrt( preR ) * epsRegul;
	if( !( R >= 0 ) )
	{
		SG_PRINT("MKL-direct: p = %.3f\n", 1.0 )
		SG_PRINT("MKL-direct: nofKernelsGood = %d\n", nofKernelsGood )
		SG_PRINT("MKL-direct: Z = %e\n", Z )
		SG_PRINT("MKL-direct: eps = %e\n", epsRegul )
		for( p=0; p<num_kernels; ++p )
		{
			const float64_t t = CMath::pow( beta_local[p] - beta[p], 2.0 );
			SG_PRINT("MKL-direct: t[%3d] = %e  ( diff = %e = %e - %e )\n", p, t, beta_local[p]-beta[p], beta_local[p], beta[p] )
		}
		SG_PRINT("MKL-direct: preR = %e\n", preR )
		SG_PRINT("MKL-direct: preR/p = %e\n", preR )
		SG_PRINT("MKL-direct: sqrt(preR/p) = %e\n", CMath::sqrt(preR) )
		SG_PRINT("MKL-direct: R = %e\n", R )
		SG_ERROR("Assertion R >= 0 failed!\n" )
	}

	Z = 0.0;
	for( p=0; p<num_kernels; ++p )
	{
		beta[p] += R;
		Z += beta[p];
		ASSERT( beta[p] >= 0 )
	}
	Z = CMath::pow( Z, -1.0 );
	ASSERT( Z >= 0 )
	for( p=0; p<num_kernels; ++p )
	{
		beta[p] *= Z;
		ASSERT( beta[p] >= 0.0 )
		if (beta[p] > 1.0 )
			beta[p] = 1.0;
	}

	// --- copy beta into beta_local
	for( p=0; p<num_kernels; ++p )
		beta_local[p] = beta[p];

	// --- elastic-net transform
	elasticnet_transform(beta, ent_lambda, num_kernels);

	// --- objective
	obj = -suma;
	for (p=0; p<num_kernels; ++p )
	{
		//obj += sumw[p] * old_beta[p]*old_beta[p] / beta[p];
		obj += sumw[p] * beta[p];
	}
	return obj;
}

void CMKL::elasticnet_dual(float64_t *ff, float64_t *gg, float64_t *hh,
		const float64_t &del, const float64_t* nm, int32_t len,
		const float64_t &lambda)
{
	std::list<int32_t> I;
	float64_t gam = 1.0-lambda;
	for (int32_t i=0; i<len;i++)
	{
		if (nm[i]>=CMath::sqrt(2*del*gam))
			I.push_back(i);
	}
	int32_t M=I.size();

	*ff=del;
	*gg=-(M*gam+lambda);
	*hh=0;
	for (std::list<int32_t>::iterator it=I.begin(); it!=I.end(); it++)
	{
		float64_t nmit = nm[*it];

		*ff += del*gam*CMath::pow(nmit/CMath::sqrt(2*del*gam)-1,2)/lambda;
		*gg += CMath::sqrt(gam/(2*del))*nmit;
		*hh += -0.5*CMath::sqrt(gam/(2*CMath::pow(del,3)))*nmit;
	}
}

// assumes that all constraints are satisfied
float64_t CMKL::compute_elasticnet_dual_objective()
{
	int32_t n=get_num_support_vectors();
	int32_t num_kernels = kernel->get_num_subkernels();
	float64_t mkl_obj=0;

	if (m_labels && kernel && kernel->get_kernel_type() == K_COMBINED)
	{
		// Compute Elastic-net dual
		float64_t* nm = SG_MALLOC(float64_t, num_kernels);
		float64_t del=0;


		int32_t k=0;
		for (index_t k_idx=0; k_idx<((CCombinedKernel*) kernel)->get_num_kernels(); k_idx++)
		{
			CKernel* kn = ((CCombinedKernel*) kernel)->get_kernel(k_idx);
			float64_t sum=0;
			for (int32_t i=0; i<n; i++)
			{
				int32_t ii=get_support_vector(i);

				for (int32_t j=0; j<n; j++)
				{
					int32_t jj=get_support_vector(j);
					sum+=get_alpha(i)*get_alpha(j)*kn->kernel(ii,jj);
				}
			}
			nm[k]= CMath::pow(sum, 0.5);
			del = CMath::max(del, nm[k]);

			// SG_PRINT("nm[%d]=%f\n",k,nm[k])
			k++;


			SG_UNREF(kn);
		}
		// initial delta
		del = del/CMath::sqrt(2*(1-ent_lambda));

		// Newton's method to optimize delta
		k=0;
		float64_t ff, gg, hh;
		elasticnet_dual(&ff, &gg, &hh, del, nm, num_kernels, ent_lambda);
		while (CMath::abs(gg)>+1e-8 && k<100)
		{
			float64_t ff_old = ff;
			float64_t gg_old = gg;
			float64_t del_old = del;

			// SG_PRINT("[%d] fval=%f gg=%f del=%f\n", k, ff, gg, del)

			float64_t step=1.0;
			do
			{
				del=del_old*CMath::exp(-step*gg/(hh*del+gg));
				elasticnet_dual(&ff, &gg, &hh, del, nm, num_kernels, ent_lambda);
				step/=2;
			} while(ff>ff_old+1e-4*gg_old*(del-del_old));

			k++;
		}
		mkl_obj=-ff;

		SG_FREE(nm);

		mkl_obj+=compute_sum_alpha();

	}
	else
		SG_ERROR("cannot compute objective, labels or kernel not set\n")

	return -mkl_obj;
}

float64_t CMKL::compute_optimal_betas_block_norm(
  float64_t* beta, const float64_t* old_beta, const int32_t num_kernels,
  const float64_t* sumw, const float64_t suma,
  const float64_t mkl_objective )
{
	float64_t obj;
	float64_t Z=0;
	int32_t p;

	// --- optimal beta
	for( p=0; p<num_kernels; ++p )
	{
		ASSERT(sumw[p]>=0)

		beta[p] = CMath::pow( sumw[p], -(2.0-mkl_block_norm)/(2.0-2.0*mkl_block_norm));
		Z+= CMath::pow( sumw[p], -(mkl_block_norm)/(2.0-2.0*mkl_block_norm));

		ASSERT( beta[p] >= 0 )
	}

	ASSERT(Z>=0)

	Z=1.0/CMath::pow(Z, (2.0-mkl_block_norm)/mkl_block_norm);

	for( p=0; p<num_kernels; ++p )
		beta[p] *= Z;

	// --- objective
	obj = -suma;
	for( p=0; p<num_kernels; ++p )
		obj += sumw[p] * beta[p];

	return obj;
}


float64_t CMKL::compute_optimal_betas_directly(
  float64_t* beta, const float64_t* old_beta, const int32_t num_kernels,
  const float64_t* sumw, const float64_t suma,
  const float64_t mkl_objective )
{
	const float64_t epsRegul = 0.01;  // fraction of root mean squared deviation
	float64_t obj;
	float64_t Z;
	float64_t preR;
	int32_t p;
	int32_t nofKernelsGood;

	// --- optimal beta
	nofKernelsGood = num_kernels;
	for( p=0; p<num_kernels; ++p )
	{
		//SG_PRINT("MKL-direct:  sumw[%3d] = %e  ( oldbeta = %e )\n", p, sumw[p], old_beta[p] )
		if( sumw[p] >= 0.0 && old_beta[p] >= 0.0 )
		{
			beta[p] = sumw[p] * old_beta[p]*old_beta[p] / mkl_norm;
			beta[p] = CMath::pow( beta[p], 1.0 / (mkl_norm+1.0) );
		}
		else
		{
			beta[p] = 0.0;
			--nofKernelsGood;
		}
		ASSERT( beta[p] >= 0 )
	}

	// --- normalize
	Z = 0.0;
	for( p=0; p<num_kernels; ++p )
		Z += CMath::pow( beta[p], mkl_norm );

	Z = CMath::pow( Z, -1.0/mkl_norm );
	ASSERT( Z >= 0 )
	for( p=0; p<num_kernels; ++p )
		beta[p] *= Z;

	// --- regularize & renormalize
	preR = 0.0;
	for( p=0; p<num_kernels; ++p )
		preR += CMath::sq( old_beta[p] - beta[p]);

	const float64_t R = CMath::sqrt( preR / mkl_norm ) * epsRegul;
	if( !( R >= 0 ) )
	{
		SG_PRINT("MKL-direct: p = %.3f\n", mkl_norm )
		SG_PRINT("MKL-direct: nofKernelsGood = %d\n", nofKernelsGood )
		SG_PRINT("MKL-direct: Z = %e\n", Z )
		SG_PRINT("MKL-direct: eps = %e\n", epsRegul )
		for( p=0; p<num_kernels; ++p )
		{
			const float64_t t = CMath::pow( old_beta[p] - beta[p], 2.0 );
			SG_PRINT("MKL-direct: t[%3d] = %e  ( diff = %e = %e - %e )\n", p, t, old_beta[p]-beta[p], old_beta[p], beta[p] )
		}
		SG_PRINT("MKL-direct: preR = %e\n", preR )
		SG_PRINT("MKL-direct: preR/p = %e\n", preR/mkl_norm )
		SG_PRINT("MKL-direct: sqrt(preR/p) = %e\n", CMath::sqrt(preR/mkl_norm) )
		SG_PRINT("MKL-direct: R = %e\n", R )
		SG_ERROR("Assertion R >= 0 failed!\n" )
	}

	Z = 0.0;
	for( p=0; p<num_kernels; ++p )
	{
		beta[p] += R;
		Z += CMath::pow( beta[p], mkl_norm );
		ASSERT( beta[p] >= 0 )
	}
	Z = CMath::pow( Z, -1.0/mkl_norm );
	ASSERT( Z >= 0 )
	for( p=0; p<num_kernels; ++p )
	{
		beta[p] *= Z;
		ASSERT( beta[p] >= 0.0 )
		if( beta[p] > 1.0 )
			beta[p] = 1.0;
	}

	// --- objective
	obj = -suma;
	for( p=0; p<num_kernels; ++p )
		obj += sumw[p] * beta[p];

	return obj;
}

float64_t CMKL::compute_optimal_betas_newton(float64_t* beta,
		const float64_t* old_beta, int32_t num_kernels,
		const float64_t* sumw, float64_t suma,
		 float64_t mkl_objective)
{
	SG_DEBUG("MKL via NEWTON\n")

	if (mkl_norm==1)
		SG_ERROR("MKL via NEWTON works only for norms>1\n")

	const double epsBeta = 1e-32;
	const double epsGamma = 1e-12;
	const double epsWsq = 1e-12;
	const double epsNewt = 0.0001;
	const double epsStep = 1e-9;
	const int nofNewtonSteps = 3;
	const double hessRidge = 1e-6;
	const int inLogSpace = 0;

	const float64_t r = mkl_norm / ( mkl_norm - 1.0 );
	float64_t* newtDir = SG_MALLOC(float64_t,  num_kernels );
	float64_t* newtBeta = SG_MALLOC(float64_t,  num_kernels );
	//float64_t newtStep;
	float64_t stepSize;
	float64_t Z;
	float64_t obj;
	float64_t gamma;
	int32_t p;
	int i;

	// --- check beta
	Z = 0.0;
	for( p=0; p<num_kernels; ++p )
	{
		beta[p] = old_beta[p];
		if( !( beta[p] >= epsBeta ) )
			beta[p] = epsBeta;

		ASSERT( 0.0 <= beta[p] && beta[p] <= 1.0 )
		Z += CMath::pow( beta[p], mkl_norm );
	}

	Z = CMath::pow( Z, -1.0/mkl_norm );
	if( !( fabs(Z-1.0) <= epsGamma ) )
	{
		SG_WARNING("old_beta not normalized (diff=%e);  forcing normalization.  ", Z-1.0 )
		for( p=0; p<num_kernels; ++p )
		{
			beta[p] *= Z;
			if( beta[p] > 1.0 )
				beta[p] = 1.0;
			ASSERT( 0.0 <= beta[p] && beta[p] <= 1.0 )
		}
	}

	// --- compute gamma
	gamma = 0.0;
	for ( p=0; p<num_kernels; ++p )
	{
		if ( !( sumw[p] >= 0 ) )
		{
			if( !( sumw[p] >= -epsWsq ) )
				SG_WARNING("sumw[%d] = %e;  treated as 0.  ", p, sumw[p] )
			// should better recompute sumw[] !!!
		}
		else
		{
			ASSERT( sumw[p] >= 0 )
			//gamma += CMath::pow( sumw[p] * beta[p]*beta[p], r );
			gamma += CMath::pow( sumw[p] * beta[p]*beta[p] / mkl_norm, r );
		}
	}
	gamma = CMath::pow( gamma, 1.0/r ) / mkl_norm;
	ASSERT( gamma > -1e-9 )
	if( !( gamma > epsGamma ) )
	{
		SG_WARNING("bad gamma: %e;  set to %e.  ", gamma, epsGamma )
		// should better recompute sumw[] !!!
		gamma = epsGamma;
	}
	ASSERT( gamma >= epsGamma )
	//gamma = -gamma;

	// --- compute objective
	obj = 0.0;
	for( p=0; p<num_kernels; ++p )
	{
		obj += beta[p] * sumw[p];
		//obj += gamma/mkl_norm * CMath::pow( beta[p], mkl_norm );
	}
	if( !( obj >= 0.0 ) )
		SG_WARNING("negative objective: %e.  ", obj )
	//SG_PRINT("OBJ = %e.  \n", obj )


	// === perform Newton steps
	for (i = 0; i < nofNewtonSteps; ++i )
	{
		// --- compute Newton direction (Hessian is diagonal)
		const float64_t gqq1 = mkl_norm * (mkl_norm-1.0) * gamma;
		// newtStep = 0.0;
		for( p=0; p<num_kernels; ++p )
		{
			ASSERT( 0.0 <= beta[p] && beta[p] <= 1.0 )
			//const float halfw2p = ( sumw[p] >= 0.0 ) ? sumw[p] : 0.0;
			//const float64_t t1 = halfw2p*beta[p] - mkl_norm*gamma*CMath::pow(beta[p],mkl_norm);
			//const float64_t t2 = 2.0*halfw2p + gqq1*CMath::pow(beta[p],mkl_norm-1.0);
			const float halfw2p = ( sumw[p] >= 0.0 ) ? (sumw[p]*old_beta[p]*old_beta[p]) : 0.0;
			const float64_t t0 = halfw2p*beta[p] - mkl_norm*gamma*CMath::pow(beta[p],mkl_norm+2.0);
			const float64_t t1 = ( t0 < 0 ) ? 0.0 : t0;
			const float64_t t2 = 2.0*halfw2p + gqq1*CMath::pow(beta[p],mkl_norm+1.0);
			if( inLogSpace )
				newtDir[p] = t1 / ( t1 + t2*beta[p] + hessRidge );
			else
				newtDir[p] = ( t1 == 0.0 ) ? 0.0 : ( t1 / t2 );
			// newtStep += newtDir[p] * grad[p];
			ASSERT( newtDir[p] == newtDir[p] )
			//SG_PRINT("newtDir[%d] = %6.3f = %e / %e \n", p, newtDir[p], t1, t2 )
		}
		//CMath::display_vector( newtDir, num_kernels, "newton direction  " );
		//SG_PRINT("Newton step size = %e\n", Z )

		// --- line search
		stepSize = 1.0;
		while( stepSize >= epsStep )
		{
			// --- perform Newton step
			Z = 0.0;
			while( Z == 0.0 )
			{
				for( p=0; p<num_kernels; ++p )
				{
					if( inLogSpace )
						newtBeta[p] = beta[p] * CMath::exp( + stepSize * newtDir[p] );
					else
						newtBeta[p] = beta[p] + stepSize * newtDir[p];
					if( !( newtBeta[p] >= epsBeta ) )
						newtBeta[p] = epsBeta;
					Z += CMath::pow( newtBeta[p], mkl_norm );
				}
				ASSERT( 0.0 <= Z )
				Z = CMath::pow( Z, -1.0/mkl_norm );
				if( Z == 0.0 )
					stepSize /= 2.0;
			}

			// --- normalize new beta (wrt p-norm)
			ASSERT( 0.0 < Z )
			ASSERT( Z < CMath::INFTY )
			for( p=0; p<num_kernels; ++p )
			{
				newtBeta[p] *= Z;
				if( newtBeta[p] > 1.0 )
				{
					//SG_WARNING("beta[%d] = %e;  set to 1.  ", p, beta[p] )
					newtBeta[p] = 1.0;
				}
				ASSERT( 0.0 <= newtBeta[p] && newtBeta[p] <= 1.0 )
			}

			// --- objective increased?
			float64_t newtObj;
			newtObj = 0.0;
			for( p=0; p<num_kernels; ++p )
				newtObj += sumw[p] * old_beta[p]*old_beta[p] / newtBeta[p];
			//SG_PRINT("step = %.8f:  obj = %e -> %e.  \n", stepSize, obj, newtObj )
			if ( newtObj < obj - epsNewt*stepSize*obj )
			{
				for( p=0; p<num_kernels; ++p )
					beta[p] = newtBeta[p];
				obj = newtObj;
				break;
			}
			stepSize /= 2.0;

		}

		if( stepSize < epsStep )
			break;
	}
	SG_FREE(newtDir);
	SG_FREE(newtBeta);

	// === return new objective
	obj = -suma;
	for( p=0; p<num_kernels; ++p )
		obj += beta[p] * sumw[p];
	return obj;
}



float64_t CMKL::compute_optimal_betas_via_cplex(float64_t* new_beta, const float64_t* old_beta, int32_t num_kernels,
		  const float64_t* sumw, float64_t suma, int32_t& inner_iters)
{
	SG_DEBUG("MKL via CPLEX\n")

#ifdef USE_CPLEX
	ASSERT(new_beta)
	ASSERT(old_beta)

	int32_t NUMCOLS = 2*num_kernels + 1;
	double* x=SG_MALLOC(double, NUMCOLS);

	if (!lp_initialized)
	{
		SG_INFO("creating LP\n")

		double   obj[NUMCOLS]; /* calling external lib */
		double   lb[NUMCOLS]; /* calling external lib */
		double   ub[NUMCOLS]; /* calling external lib */

		for (int32_t i=0; i<2*num_kernels; i++)
		{
			obj[i]=0 ;
			lb[i]=0 ;
			ub[i]=1 ;
		}

		for (int32_t i=num_kernels; i<2*num_kernels; i++)
			obj[i]= C_mkl;

		obj[2*num_kernels]=1 ;
		lb[2*num_kernels]=-CPX_INFBOUND ;
		ub[2*num_kernels]=CPX_INFBOUND ;

		int status = CPXnewcols (env, lp_cplex, NUMCOLS, obj, lb, ub, NULL, NULL);
		if ( status ) {
			char  errmsg[1024];
			CPXgeterrorstring (env, status, errmsg);
			SG_ERROR("%s", errmsg)
		}

		// add constraint sum(w)=1;
		SG_INFO("adding the first row\n")
		int initial_rmatbeg[1]; /* calling external lib */
		int initial_rmatind[num_kernels+1]; /* calling external lib */
		double initial_rmatval[num_kernels+1]; /* calling ext lib */
		double initial_rhs[1]; /* calling external lib */
		char initial_sense[1];

		// 1-norm MKL
		if (mkl_norm==1)
		{
			initial_rmatbeg[0] = 0;
			initial_rhs[0]=1 ;     // rhs=1 ;
			initial_sense[0]='E' ; // equality

			// sparse matrix
			for (int32_t i=0; i<num_kernels; i++)
			{
				initial_rmatind[i]=i ; //index of non-zero element
				initial_rmatval[i]=1 ; //value of ...
			}
			initial_rmatind[num_kernels]=2*num_kernels ; //number of non-zero elements
			initial_rmatval[num_kernels]=0 ;

			status = CPXaddrows (env, lp_cplex, 0, 1, num_kernels+1,
					initial_rhs, initial_sense, initial_rmatbeg,
					initial_rmatind, initial_rmatval, NULL, NULL);

		}
		else // 2 and q-norm MKL
		{
			initial_rmatbeg[0] = 0;
			initial_rhs[0]=0 ;     // rhs=1 ;
			initial_sense[0]='L' ; // <=  (inequality)

			initial_rmatind[0]=2*num_kernels ;
			initial_rmatval[0]=0 ;

			status = CPXaddrows (env, lp_cplex, 0, 1, 1,
					initial_rhs, initial_sense, initial_rmatbeg,
					initial_rmatind, initial_rmatval, NULL, NULL);


			if (mkl_norm==2)
			{
				for (int32_t i=0; i<num_kernels; i++)
				{
					initial_rmatind[i]=i ;
					initial_rmatval[i]=1 ;
				}
				initial_rmatind[num_kernels]=2*num_kernels ;
				initial_rmatval[num_kernels]=0 ;

				status = CPXaddqconstr (env, lp_cplex, 0, num_kernels+1, 1.0, 'L', NULL, NULL,
						initial_rmatind, initial_rmatind, initial_rmatval, NULL);
			}
		}


		if ( status )
			SG_ERROR("Failed to add the first row.\n")

		lp_initialized = true ;

		if (C_mkl!=0.0)
		{
			for (int32_t q=0; q<num_kernels-1; q++)
			{
				// add constraint w[i]-w[i+1]<s[i];
				// add constraint w[i+1]-w[i]<s[i];
				int rmatbeg[1]; /* calling external lib */
				int rmatind[3]; /* calling external lib */
				double rmatval[3]; /* calling external lib */
				double rhs[1]; /* calling external lib */
				char sense[1];

				rmatbeg[0] = 0;
				rhs[0]=0 ;     // rhs=0 ;
				sense[0]='L' ; // <=
				rmatind[0]=q ;
				rmatval[0]=1 ;
				rmatind[1]=q+1 ;
				rmatval[1]=-1 ;
				rmatind[2]=num_kernels+q ;
				rmatval[2]=-1 ;
				status = CPXaddrows (env, lp_cplex, 0, 1, 3,
						rhs, sense, rmatbeg,
						rmatind, rmatval, NULL, NULL);
				if ( status )
					SG_ERROR("Failed to add a smothness row (1).\n")

				rmatbeg[0] = 0;
				rhs[0]=0 ;     // rhs=0 ;
				sense[0]='L' ; // <=
				rmatind[0]=q ;
				rmatval[0]=-1 ;
				rmatind[1]=q+1 ;
				rmatval[1]=1 ;
				rmatind[2]=num_kernels+q ;
				rmatval[2]=-1 ;
				status = CPXaddrows (env, lp_cplex, 0, 1, 3,
						rhs, sense, rmatbeg,
						rmatind, rmatval, NULL, NULL);
				if ( status )
					SG_ERROR("Failed to add a smothness row (2).\n")
			}
		}
	}

	{ // add the new row
		//SG_INFO("add the new row\n")

		int rmatbeg[1];
		int rmatind[num_kernels+1];
		double rmatval[num_kernels+1];
		double rhs[1];
		char sense[1];

		rmatbeg[0] = 0;
		if (mkl_norm==1)
			rhs[0]=0 ;
		else
			rhs[0]=-suma ;

		sense[0]='L' ;

		for (int32_t i=0; i<num_kernels; i++)
		{
			rmatind[i]=i ;
			if (mkl_norm==1)
				rmatval[i]=-(sumw[i]-suma) ;
			else
				rmatval[i]=-sumw[i];
		}
		rmatind[num_kernels]=2*num_kernels ;
		rmatval[num_kernels]=-1 ;

		int32_t status = CPXaddrows (env, lp_cplex, 0, 1, num_kernels+1,
				rhs, sense, rmatbeg,
				rmatind, rmatval, NULL, NULL);
		if ( status )
			SG_ERROR("Failed to add the new row.\n")
	}

	inner_iters=0;
	int status;
	{

		if (mkl_norm==1) // optimize 1 norm MKL
			status = CPXlpopt (env, lp_cplex);
		else if (mkl_norm==2) // optimize 2-norm MKL
			status = CPXbaropt(env, lp_cplex);
		else // q-norm MKL
		{
			float64_t* beta=SG_MALLOC(float64_t, 2*num_kernels+1);
			float64_t objval_old=1e-8; //some value to cause the loop to not terminate yet
			for (int32_t i=0; i<num_kernels; i++)
				beta[i]=old_beta[i];
			for (int32_t i=num_kernels; i<2*num_kernels+1; i++)
				beta[i]=0;

			while (true)
			{
				//int rows=CPXgetnumrows(env, lp_cplex);
				//int cols=CPXgetnumcols(env, lp_cplex);
				//SG_PRINT("rows:%d, cols:%d (kernel:%d)\n", rows, cols, num_kernels)
				CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

				set_qnorm_constraints(beta, num_kernels);

				status = CPXbaropt(env, lp_cplex);
				if ( status )
					SG_ERROR("Failed to optimize Problem.\n")

				int solstat=0;
				double objval=0;
				status=CPXsolution(env, lp_cplex, &solstat, &objval,
						(double*) beta, NULL, NULL, NULL);

				if ( status )
				{
					CMath::display_vector(beta, num_kernels, "beta");
					SG_ERROR("Failed to obtain solution.\n")
				}

				CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

				//SG_PRINT("[%d] %f (%f)\n", inner_iters, objval, objval_old)
				if ((1-abs(objval/objval_old) < 0.1*mkl_epsilon)) // && (inner_iters>2))
					break;

				objval_old=objval;

				inner_iters++;
			}
			SG_FREE(beta);
		}

		if ( status )
			SG_ERROR("Failed to optimize Problem.\n")

		// obtain solution
		int32_t cur_numrows=(int32_t) CPXgetnumrows(env, lp_cplex);
		int32_t cur_numcols=(int32_t) CPXgetnumcols(env, lp_cplex);
		int32_t num_rows=cur_numrows;
		ASSERT(cur_numcols<=2*num_kernels+1)

		float64_t* slack=SG_MALLOC(float64_t, cur_numrows);
		float64_t* pi=SG_MALLOC(float64_t, cur_numrows);

		/* calling external lib */
		int solstat=0;
		double objval=0;

		if (mkl_norm==1)
		{
			status=CPXsolution(env, lp_cplex, &solstat, &objval,
					(double*) x, (double*) pi, (double*) slack, NULL);
		}
		else
		{
			status=CPXsolution(env, lp_cplex, &solstat, &objval,
					(double*) x, NULL, (double*) slack, NULL);
		}

		int32_t solution_ok = (!status) ;
		if ( status )
			SG_ERROR("Failed to obtain solution.\n")

		int32_t num_active_rows=0 ;
		if (solution_ok)
		{
			/* 1 norm mkl */
			float64_t max_slack = -CMath::INFTY ;
			int32_t max_idx = -1 ;
			int32_t start_row = 1 ;
			if (C_mkl!=0.0)
				start_row+=2*(num_kernels-1);

			for (int32_t i = start_row; i < cur_numrows; i++)  // skip first
			{
				if (mkl_norm==1)
				{
					if ((pi[i]!=0))
						num_active_rows++ ;
					else
					{
						if (slack[i]>max_slack)
						{
							max_slack=slack[i] ;
							max_idx=i ;
						}
					}
				}
				else // 2-norm or general q-norm
				{
					if ((CMath::abs(slack[i])<1e-6))
						num_active_rows++ ;
					else
					{
						if (slack[i]>max_slack)
						{
							max_slack=slack[i] ;
							max_idx=i ;
						}
					}
				}
			}

			// have at most max(100,num_active_rows*2) rows, if not, remove one
			if ( (num_rows-start_row>CMath::max(100,2*num_active_rows)) && (max_idx!=-1))
			{
				//SG_INFO("-%i(%i,%i)",max_idx,start_row,num_rows)
				status = CPXdelrows (env, lp_cplex, max_idx, max_idx) ;
				if ( status )
					SG_ERROR("Failed to remove an old row.\n")
			}

			//CMath::display_vector(x, num_kernels, "beta");

			rho = -x[2*num_kernels] ;
			SG_FREE(pi);
			SG_FREE(slack);

		}
		else
		{
			/* then something is wrong and we rather
			stop sooner than later */
			rho = 1 ;
		}
	}
	for (int32_t i=0; i<num_kernels; i++)
		new_beta[i]=x[i];

	SG_FREE(x);
#else
	SG_ERROR("Cplex not enabled at compile time\n")
#endif
	return rho;
}

float64_t CMKL::compute_optimal_betas_via_glpk(float64_t* beta, const float64_t* old_beta,
		int num_kernels, const float64_t* sumw, float64_t suma, int32_t& inner_iters)
{
	SG_DEBUG("MKL via GLPK\n")

	if (mkl_norm!=1)
		SG_ERROR("MKL via GLPK works only for norm=1\n")

	float64_t obj=1.0;
#ifdef USE_GLPK
	int32_t NUMCOLS = 2*num_kernels + 1 ;
	if (!lp_initialized)
	{
		SG_INFO("creating LP\n")

		//set obj function, note glpk indexing is 1-based
		glp_add_cols(lp_glpk, NUMCOLS);
		for (int i=1; i<=2*num_kernels; i++)
		{
			glp_set_obj_coef(lp_glpk, i, 0);
			glp_set_col_bnds(lp_glpk, i, GLP_DB, 0, 1);
		}
		for (int i=num_kernels+1; i<=2*num_kernels; i++)
		{
			glp_set_obj_coef(lp_glpk, i, C_mkl);
		}
		glp_set_obj_coef(lp_glpk, NUMCOLS, 1);
		glp_set_col_bnds(lp_glpk, NUMCOLS, GLP_FR, 0,0); //unbound

		//add first row. sum[w]=1
		int row_index = glp_add_rows(lp_glpk, 1);
		int* ind = SG_MALLOC(int, num_kernels+2);
		float64_t* val = SG_MALLOC(float64_t, num_kernels+2);
		for (int i=1; i<=num_kernels; i++)
		{
			ind[i] = i;
			val[i] = 1;
		}
		ind[num_kernels+1] = NUMCOLS;
		val[num_kernels+1] = 0;
		glp_set_mat_row(lp_glpk, row_index, num_kernels, ind, val);
		glp_set_row_bnds(lp_glpk, row_index, GLP_FX, 1, 1);
		SG_FREE(val);
		SG_FREE(ind);

		lp_initialized = true;

		if (C_mkl!=0.0)
		{
			for (int32_t q=1; q<num_kernels; q++)
			{
				int mat_ind[4];
				float64_t mat_val[4];
				int mat_row_index = glp_add_rows(lp_glpk, 2);
				mat_ind[1] = q;
				mat_val[1] = 1;
				mat_ind[2] = q+1;
				mat_val[2] = -1;
				mat_ind[3] = num_kernels+q;
				mat_val[3] = -1;
				glp_set_mat_row(lp_glpk, mat_row_index, 3, mat_ind, mat_val);
				glp_set_row_bnds(lp_glpk, mat_row_index, GLP_UP, 0, 0);
				mat_val[1] = -1;
				mat_val[2] = 1;
				glp_set_mat_row(lp_glpk, mat_row_index+1, 3, mat_ind, mat_val);
				glp_set_row_bnds(lp_glpk, mat_row_index+1, GLP_UP, 0, 0);
			}
		}
	}

	int* ind=SG_MALLOC(int,num_kernels+2);
	float64_t* val=SG_MALLOC(float64_t, num_kernels+2);
	int row_index = glp_add_rows(lp_glpk, 1);
	for (int32_t i=1; i<=num_kernels; i++)
	{
		ind[i] = i;
		val[i] = -(sumw[i-1]-suma);
	}
	ind[num_kernels+1] = 2*num_kernels+1;
	val[num_kernels+1] = -1;
	glp_set_mat_row(lp_glpk, row_index, num_kernels+1, ind, val);
	glp_set_row_bnds(lp_glpk, row_index, GLP_UP, 0, 0);
	SG_FREE(ind);
	SG_FREE(val);

	//optimize
	glp_simplex(lp_glpk, lp_glpk_parm);
	bool res = check_glp_status(lp_glpk);
	if (!res)
		SG_ERROR("Failed to optimize Problem.\n")

	int32_t cur_numrows = glp_get_num_rows(lp_glpk);
	int32_t cur_numcols = glp_get_num_cols(lp_glpk);
	int32_t num_rows=cur_numrows;
	ASSERT(cur_numcols<=2*num_kernels+1)

	float64_t* col_primal = SG_MALLOC(float64_t, cur_numcols);
	float64_t* row_primal = SG_MALLOC(float64_t, cur_numrows);
	float64_t* row_dual = SG_MALLOC(float64_t, cur_numrows);

	for (int i=0; i<cur_numrows; i++)
	{
		row_primal[i] = glp_get_row_prim(lp_glpk, i+1);
		row_dual[i] = glp_get_row_dual(lp_glpk, i+1);
	}
	for (int i=0; i<cur_numcols; i++)
		col_primal[i] = glp_get_col_prim(lp_glpk, i+1);

	obj = -col_primal[2*num_kernels];

	for (int i=0; i<num_kernels; i++)
		beta[i] = col_primal[i];

	int32_t num_active_rows=0;
	if(res)
	{
		float64_t max_slack = CMath::INFTY;
		int32_t max_idx = -1;
		int32_t start_row = 1;
		if (C_mkl!=0.0)
			start_row += 2*(num_kernels-1);

		for (int32_t i= start_row; i<cur_numrows; i++)
		{
			if (row_dual[i]!=0)
				num_active_rows++;
			else
			{
				if (row_primal[i]<max_slack)
				{
					max_slack = row_primal[i];
					max_idx = i;
				}
			}
		}

		if ((num_rows-start_row>CMath::max(100, 2*num_active_rows)) && max_idx!=-1)
		{
			int del_rows[2];
			del_rows[1] = max_idx+1;
			glp_del_rows(lp_glpk, 1, del_rows);
		}
	}

	SG_FREE(row_dual);
	SG_FREE(row_primal);
	SG_FREE(col_primal);
#else
	SG_ERROR("Glpk not enabled at compile time\n")
#endif

	return obj;
}

void CMKL::compute_sum_beta(float64_t* sumw)
{
	ASSERT(sumw)
	ASSERT(svm)

	int32_t nsv=svm->get_num_support_vectors();
	int32_t num_kernels = kernel->get_num_subkernels();
	SGVector<float64_t> beta=SGVector<float64_t>(num_kernels);
	int32_t nweights=0;
	const float64_t* old_beta = kernel->get_subkernel_weights(nweights);
	ASSERT(nweights==num_kernels)
	ASSERT(old_beta)

	for (int32_t i=0; i<num_kernels; i++)
	{
		beta.vector[i]=0;
		sumw[i]=0;
	}

	for (int32_t n=0; n<num_kernels; n++)
	{
		beta.vector[n]=1.0;
		/* this only copies the value of the first entry of this array
		 * so it may be freed safely afterwards. */
		kernel->set_subkernel_weights(beta);

		for (int32_t i=0; i<nsv; i++)
		{
			int32_t ii=svm->get_support_vector(i);

			for (int32_t j=0; j<nsv; j++)
			{
				int32_t jj=svm->get_support_vector(j);
				sumw[n]+=0.5*svm->get_alpha(i)*svm->get_alpha(j)*kernel->kernel(ii,jj);
			}
		}
		beta[n]=0.0;
	}

	mkl_iterations++;
	kernel->set_subkernel_weights(SGVector<float64_t>( (float64_t*) old_beta, num_kernels, false));
}


// assumes that all constraints are satisfied
float64_t CMKL::compute_mkl_dual_objective()
{
	if (get_solver_type()==ST_ELASTICNET)
	{
		// -- Compute ElasticnetMKL dual objective
		return compute_elasticnet_dual_objective();
	}

	int32_t n=get_num_support_vectors();
	float64_t mkl_obj=0;

	if (m_labels && kernel && kernel->get_kernel_type() == K_COMBINED)
	{
		for (index_t k_idx=0; k_idx<((CCombinedKernel*) kernel)->get_num_kernels(); k_idx++)
		{
			CKernel* kn = ((CCombinedKernel*) kernel)->get_kernel(k_idx);
			float64_t sum=0;
			for (int32_t i=0; i<n; i++)
			{
				int32_t ii=get_support_vector(i);

				for (int32_t j=0; j<n; j++)
				{
					int32_t jj=get_support_vector(j);
					sum+=get_alpha(i)*get_alpha(j)*kn->kernel(ii,jj);
				}
			}

			if (mkl_norm==1.0)
				mkl_obj = CMath::max(mkl_obj, sum);
			else
				mkl_obj += CMath::pow(sum, mkl_norm/(mkl_norm-1));

			SG_UNREF(kn);
		}

		if (mkl_norm==1.0)
			mkl_obj=-0.5*mkl_obj;
		else
			mkl_obj= -0.5*CMath::pow(mkl_obj, (mkl_norm-1)/mkl_norm);

		mkl_obj+=compute_sum_alpha();
	}
	else
		SG_ERROR("cannot compute objective, labels or kernel not set\n")

	return -mkl_obj;
}

#ifdef USE_CPLEX
void CMKL::set_qnorm_constraints(float64_t* beta, int32_t num_kernels)
{
	ASSERT(num_kernels>0)

	float64_t* grad_beta=SG_MALLOC(float64_t, num_kernels);
	float64_t* hess_beta=SG_MALLOC(float64_t, num_kernels+1);
	float64_t* lin_term=SG_MALLOC(float64_t, num_kernels+1);
	int* ind=SG_MALLOC(int, num_kernels+1);

	//CMath::display_vector(beta, num_kernels, "beta");
	double const_term = 1-CMath::qsq(beta, num_kernels, mkl_norm);

	//SG_PRINT("const=%f\n", const_term)
	ASSERT(CMath::fequal(const_term, 0.0))

	for (int32_t i=0; i<num_kernels; i++)
	{
		grad_beta[i]=mkl_norm * pow(beta[i], mkl_norm-1);
		hess_beta[i]=0.5*mkl_norm*(mkl_norm-1) * pow(beta[i], mkl_norm-2);
		lin_term[i]=grad_beta[i] - 2*beta[i]*hess_beta[i];
		const_term+=grad_beta[i]*beta[i] - CMath::sq(beta[i])*hess_beta[i];
		ind[i]=i;
	}
	ind[num_kernels]=2*num_kernels;
	hess_beta[num_kernels]=0;
	lin_term[num_kernels]=0;

	int status=0;
	int num=CPXgetnumqconstrs (env, lp_cplex);

	if (num>0)
	{
		status = CPXdelqconstrs (env, lp_cplex, 0, 0);
		ASSERT(!status)
	}

	status = CPXaddqconstr (env, lp_cplex, num_kernels+1, num_kernels+1, const_term, 'L', ind, lin_term,
			ind, ind, hess_beta, NULL);
	ASSERT(!status)

	//CPXwriteprob (env, lp_cplex, "prob.lp", NULL);
	//CPXqpwrite (env, lp_cplex, "prob.qp");

	SG_FREE(grad_beta);
	SG_FREE(hess_beta);
	SG_FREE(lin_term);
	SG_FREE(ind);
}
#endif // USE_CPLEX

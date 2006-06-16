/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#ifdef USE_SVMLIGHT

#ifdef HAVE_PYTHON
#include <Python.h>
#endif

#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Mathmatics.h"
#include "regression/svr/SVR_light.h"
#include "classifier/svm/Optimizer.h"
#include "kernel/KernelMachine.h"
#include "kernel/CombinedKernel.h"

#include <unistd.h>


#ifdef HAVE_LAPACK
extern "C" {
#include <cblas.h>
}
#endif

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}
#endif

#include "lib/Parallel.h"

#include <unistd.h>
#include <pthread.h>

struct S_THREAD_PARAM 
{
	DREAL * lin ;
	INT start, end;
	LONG * active2dnum ;
	LONG * docs ;
	CKernel* kernel ;
    INT num_vectors ;
}  ;

CSVRLight::CSVRLight()
{
	W=NULL;
	model=new MODEL[1];
	learn_parm=new LEARN_PARM[1];
	model->supvec=NULL;
	model->alpha=NULL;
	model->index=NULL;
	set_kernel(NULL);

	//certain setup params
	verbosity=1;
	init_margin=0.15;
	init_iter=500;
	precision_violations=0;
	opt_precision=DEF_PRECISION;

	// MKL stuff
	rho=0 ;
	mymaxdiff=1 ;
	num_rows=0 ;
	num_active_rows=0 ;
	weight_epsilon=0 ;
	lp_C = 0 ;
	buffer_num      = NULL ;
	buffer_numcols  = NULL ;
	
	precomputed_subkernels = NULL ;
	num_precomputed_subkernels = 0 ;
	use_kernel_cache = true ;

#ifdef USE_CPLEX
	lp = NULL ;
	env = NULL ;
	lp_initialized = false ;
#endif
	
}

bool CSVRLight::train()
{
	//certain setup params	
	verbosity=1;
	init_margin=0.15;
	init_iter=500;
	precision_violations=0;
	opt_precision=DEF_PRECISION;
	
	strcpy (learn_parm->predfile, "");
	learn_parm->biased_hyperplane=1; 
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=1;
	learn_parm->svm_maxqpsize=get_qpsize();
	learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize-1;
	learn_parm->maxiter=100000;
	learn_parm->svm_iter_to_shrink=100;
	learn_parm->svm_c=get_C1();
	learn_parm->eps=tube_epsilon;      /* equivalent regression epsilon for classification */
	learn_parm->transduction_posratio=0.33;
	learn_parm->svm_costratio=get_C2()/get_C1();
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=epsilon; // GU: better decrease it ... ??
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;

	if (!CKernelMachine::get_kernel())
	{
		CIO::message(M_ERROR, "SVM_light can not proceed without kernel!\n");
		return false ;
	}

	if (weight_epsilon<=0)
		weight_epsilon=1e-2 ;
	if (get_kernel()->has_property(KP_LINADD) && get_linadd_enabled())
		get_kernel()->clear_normal();

	// output some info
	CIO::message(M_DEBUG, "qpsize = %i\n", learn_parm->svm_maxqpsize) ;
	CIO::message(M_DEBUG, "epsilon = %1.1e\n", learn_parm->epsilon_crit) ;
	CIO::message(M_DEBUG, "weight_epsilon = %1.1e\n", weight_epsilon) ;
	CIO::message(M_DEBUG, "C_mkl = %1.1e\n", C_mkl) ;
	CIO::message(M_DEBUG, "get_kernel()->has_property(KP_LINADD) = %i\n", get_kernel()->has_property(KP_LINADD)) ;
	CIO::message(M_DEBUG, "get_kernel()->has_property(KP_KERNCOMBINATION) = %i\n", get_kernel()->has_property(KP_KERNCOMBINATION)) ;
	CIO::message(M_DEBUG, "get_mkl_enabled() = %i\n", get_mkl_enabled()) ;
	CIO::message(M_DEBUG, "get_linadd_enabled() = %i\n", get_linadd_enabled()) ;
	CIO::message(M_DEBUG, "get_kernel()->get_num_subkernels() = %i\n", get_kernel()->get_num_subkernels()) ;
	CIO::message(M_DEBUG, "estimated time: %1.1f minutes\n", 5e-11*pow(get_kernel()->get_num_subkernels(),2.22)*pow(get_kernel()->get_rhs()->get_num_vectors(),1.68)*pow(log2(1/weight_epsilon),2.52)/60) ;

	use_kernel_cache = !(use_precomputed_subkernels || (get_kernel()->get_kernel_type() == K_CUSTOM) ||
						 (get_linadd_enabled() && get_kernel()->has_property(KP_LINADD)) ||
						 get_kernel()->get_precompute_matrix() || 
						 get_kernel()->get_precompute_subkernel_matrix()) ;

	CIO::message(M_DEBUG, "get_kernel()->get_precompute_matrix() = %i\n", get_kernel()->get_precompute_matrix()) ;
	CIO::message(M_DEBUG, "get_kernel()->get_precompute_subkernel_matrix() = %i\n", get_kernel()->get_precompute_subkernel_matrix()) ;
	CIO::message(M_DEBUG, "use_kernel_cache = %i\n", use_kernel_cache) ;

#ifdef USE_CPLEX
	cleanup_cplex();

	if (get_mkl_enabled())
		init_cplex();
#else
	if (get_mkl_enabled())
		CIO::message(M_ERROR, "CPLEX was disabled at compile-time\n");
#endif
	
	if (precomputed_subkernels != NULL)
	{
		for (INT i=0; i<num_precomputed_subkernels; i++)
			delete[] precomputed_subkernels[i] ;
		delete[] precomputed_subkernels ;
		num_precomputed_subkernels=0 ;
		precomputed_subkernels=NULL ;
	}

	if (use_precomputed_subkernels)
	{
		INT num = get_kernel()->get_rhs()->get_num_vectors() ;
		INT num_kernels = get_kernel()->get_num_subkernels() ;
		num_precomputed_subkernels=num_kernels ;
		precomputed_subkernels=new SHORTREAL*[num_precomputed_subkernels] ;
		CKernel* k = get_kernel() ;
		INT num_weights = -1;
		const DREAL* w   = k->get_subkernel_weights(num_weights);

		DREAL* w_backup = new DREAL[num_kernels] ;
		DREAL* w1 = new DREAL[num_kernels] ;
		
		// backup and set to zero
		for (INT i=0; i<num_kernels; i++)
		{
			w_backup[i] = w[i] ;
			w1[i]=0.0 ; 
		}

        // allocating memory 
		for (INT n=0; n<num_precomputed_subkernels; n++)
		{
			precomputed_subkernels[n]=new SHORTREAL[num*(num+1)/2] ;
			ASSERT(precomputed_subkernels[n]!=NULL) ;
		}
		
		for (INT n=0; n<num_precomputed_subkernels; n++)
		{
			w1[n]=1.0 ;
			k->set_subkernel_weights(w1, num_weights) ;

			SHORTREAL * matrix = precomputed_subkernels[n] ;
			
			CIO::message(M_INFO, "precomputing kernel matrix %i (%ix%i)\n", n, num, num) ;
			for (INT i=0; i<num; i++)
			{
				CIO::progress(i*i,0,num*num);
				
				for (INT j=0; j<=i; j++)
					matrix[i*(i+1)/2+j] = k->kernel(i,j) ;

			}
			CIO::progress(num*num,0,num*num);
			CIO::message(M_INFO, "\ndone.\n") ;
			w1[n]=0.0 ;
		}

		// restore old weights
		k->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;

	}

	// train the svm
	svr_learn();
	
	// brain damaged svm light work around
	create_new_model(model->sv_num-1);
	set_bias(-model->b);
	for (INT i=0; i<model->sv_num-1; i++)
	{
		set_alpha(i, model->alpha[i+1]);
		set_support_vector(i, model->supvec[i+1]);
	}

#ifdef USE_CPLEX
	cleanup_cplex();
#endif
	
	if (precomputed_subkernels!=NULL)
	{
		for (INT i=0; i<num_precomputed_subkernels; i++)
			delete[] precomputed_subkernels[i] ;
		delete[] precomputed_subkernels ;
		num_precomputed_subkernels=0 ;
		precomputed_subkernels=NULL ;
	}

	if (get_kernel()->has_property(KP_LINADD) && get_linadd_enabled())
		get_kernel()->clear_normal() ;

	return true ;
}

void CSVRLight::svr_learn()
{
	LONG *inconsistent, i, j;
	LONG inconsistentnum;
	LONG upsupvecnum;
	double maxdiff, *lin, *c, *a;
	LONG runtime_start,runtime_end;
	LONG iterations;
	double *xi_fullset; /* buffer for storing xi on full sample in loo */
	double *a_fullset;  /* buffer for storing alpha on full sample in loo */
	TIMING timing_profile;
	SHRINK_STATE shrink_state;
	INT* label;
	LONG* docs;

	CLabels* lab= CClassifier::get_labels();
	ASSERT(lab!=NULL);
	INT totdoc=lab->get_num_labels();
	num_vectors=totdoc;
	
	// set up regression problem in standard form
	docs=new long[2*totdoc];
	label=new INT[2*totdoc];
	c = new double[2*totdoc];

  for(i=0;i<totdoc;i++) {   
	  docs[i]=i;
	  j=2*totdoc-1-i;
	  label[i]=+1;
	  c[i]=lab->get_label(i);
	  docs[j]=j;
	  label[j]=-1;
	  c[j]=lab->get_label(i);
  }
  totdoc*=2;

  // MKL stuff
  buffer_num = new DREAL[totdoc];
  buffer_numcols = NULL;

  //prepare kernel cache for regression (i.e. cachelines are twice of current size)
  get_kernel()->resize_kernel_cache( get_kernel()->get_cache_size(), true);

  if ( get_kernel()->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() &&
		  (!((CCombinedKernel*)get_kernel())->get_append_subkernel_weights()) 
	 )
  {
	  CCombinedKernel* k      = (CCombinedKernel*) get_kernel();
	  CKernel* kn = k->get_first_kernel();

	  while (kn)
	  {
		  kn->resize_kernel_cache( get_kernel()->get_cache_size(), true);
		  kn = k->get_next_kernel(kn) ;
	  }
  }

  runtime_start=get_runtime();
  timing_profile.time_kernel=0;
  timing_profile.time_opti=0;
  timing_profile.time_shrink=0;
  timing_profile.time_update=0;
  timing_profile.time_model=0;
  timing_profile.time_check=0;
  timing_profile.time_select=0;

	delete[] W;
	W=NULL;
	rho=0 ;
	w_gap = 1 ;
	count = 0 ;
	num_rows=0 ;

	if (get_kernel()->has_property(KP_KERNCOMBINATION))
	{
		W = new DREAL[totdoc*get_kernel()->get_num_subkernels()];
		for (i=0; i<totdoc*get_kernel()->get_num_subkernels(); i++)
			W[i]=0;
	}

	/* make sure -n value is reasonable */
	if((learn_parm->svm_newvarsinqp < 2) 
			|| (learn_parm->svm_newvarsinqp > learn_parm->svm_maxqpsize)) {
		learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;
	}

	init_shrink_state(&shrink_state,totdoc,(LONG)MAXSHRINK);

	inconsistent = new long[totdoc];
	a = new double[totdoc];
	a_fullset = new double[totdoc];
	xi_fullset = new double[totdoc];
	lin = new double[totdoc];
	learn_parm->svm_cost = new double[totdoc];

	delete[] model->supvec;
	delete[] model->alpha;
	delete[] model->index;
	model->supvec = new long[totdoc+2];
	model->alpha = new double[totdoc+2];
	model->index = new long[totdoc+2];

	model->at_upper_bound=0;
	model->b=0;	       
	model->supvec[0]=0;  /* element 0 reserved and empty for now */
	model->alpha[0]=0;
	model->totdoc=totdoc;

	model->kernel=CKernelMachine::get_kernel();

	model->sv_num=1;
	model->loo_error=-1;
	model->loo_recall=-1;
	model->loo_precision=-1;
	model->xa_error=-1;
	model->xa_recall=-1;
	model->xa_precision=-1;
	inconsistentnum=0;

  for(i=0;i<totdoc;i++) {    /* various inits */
    inconsistent[i]=0;
    a[i]=0;
    lin[i]=0;

		if(label[i] > 0) {
			learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
				fabs((double)label[i]);
		}
		else if(label[i] < 0) {
			learn_parm->svm_cost[i]=learn_parm->svm_c*fabs((double)label[i]);
		}
		else
			ASSERT(false);
	}

	if(verbosity==1) {
		CIO::message(M_DEBUG, "Optimizing...\n");
	}

	/* train the svm */
		CIO::message(M_DEBUG, "num_train: %d\n", totdoc);
  iterations=optimize_to_convergence(docs,label,totdoc,
                     &shrink_state,model,inconsistent,a,lin,
                     c,&timing_profile,
                     &maxdiff,(long)-1,
                     (long)1);


	if(verbosity>=1) {
		CIO::message(M_INFO, "done. (%ld iterations)\n",iterations);
		CIO::message(M_INFO, "Optimization finished (maxdiff=%.8f).\n",maxdiff);
		CIO::message(M_INFO, "obj = %.16f, rho = %.16f\n",get_objective(),model->b);

		runtime_end=get_runtime();
		upsupvecnum=0;

		CIO::message(M_DEBUG, "num sv: %d\n", model->sv_num);
		for(i=1;i<model->sv_num;i++)
		{
			if(fabs(model->alpha[i]) >= 
					(learn_parm->svm_cost[model->supvec[i]]-
					 learn_parm->epsilon_a)) 
				upsupvecnum++;
		}
		CIO::message(M_INFO, "Number of SV: %ld (including %ld at upper bound)\n",
				model->sv_num-1,upsupvecnum);
	}

  /* this makes sure the model we return does not contain pointers to the 
     temporary documents */
  for(i=1;i<model->sv_num;i++) { 
    j=model->supvec[i];
    if(j >= (totdoc/2)) {
      j=totdoc-j-1;
    }
    model->supvec[i]=j;
  }
  
  shrink_state_cleanup(&shrink_state);
	delete[] label;
	delete[] inconsistent;
	delete[] c;
	delete[] a;
	delete[] a_fullset;
	delete[] xi_fullset;
	delete[] lin;
	delete[] learn_parm->svm_cost;
	delete[] docs;
	delete[] buffer_num;
	buffer_num= NULL ;
	delete[] buffer_numcols ;
	buffer_numcols = NULL ;
}

double CSVRLight::compute_objective_function(double *a, double *lin, double *c,
                  double eps, INT *label, INT totdoc)
{
  /* calculate value of objective function */
  double criterion=0;

  for(INT i=0;i<totdoc;i++)
	  criterion+=(eps-(double)label[i]*c[i])*a[i]+0.5*a[i]*label[i]*lin[i];

  /* double check=0;
  for(INT i=0;i<totdoc;i++)
  {
	  check+=a[i]*eps-a[i]*label[i]*c[i];
	  for(INT j=0;j<totdoc;j++)
		  check+= 0.5*a[i]*label[i]*a[j]*label[j]*get_kernel()->kernel(regression_fix_index(i),regression_fix_index(j));

  }

  CIO::message(M_INFO,"REGRESSION OBJECTIVE %f vs. CHECK %f (diff %f)\n", criterion, check, criterion-check); */

  return(criterion);
}


void CSVRLight::update_linear_component_mkl(LONG* docs, INT* label, 
											long int *active2dnum, double *a, 
											double *a_old, long int *working2dnum, 
											long int totdoc,
											double *lin, DREAL *aicache, double* c)
{
	CKernel* k      = get_kernel();
	INT num         = totdoc;
	INT num_weights = -1;
	INT num_kernels = k->get_num_subkernels() ;
	const DREAL* w   = k->get_subkernel_weights(num_weights);

	ASSERT(num_weights==num_kernels) ;
	DREAL* sumw = new DREAL[num_kernels];

	if (use_precomputed_subkernels) // everything is already precomputed
	{
		ASSERT(precomputed_subkernels!=NULL) ;
		for (INT n=0; n<num_kernels; n++)
		{
			ASSERT(precomputed_subkernels[n]!=NULL) ;
			SHORTREAL * matrix = precomputed_subkernels[n] ;
			for(INT ii=0;ii<num;ii++) 
			{
				if(a[ii] != a_old[ii]) 
				{
					INT i=regression_fix_index(ii);

					for(INT jj=0;jj<num;jj++) 
					{
						INT j=regression_fix_index(jj);

						if (i>=j)
							W[jj*num_kernels+n]+=(a[ii]-a_old[ii])*matrix[i*(i+1)/2+j]*(double)label[ii];
						else
							W[jj*num_kernels+n]+=(a[ii]-a_old[ii])*matrix[i+j*(j+1)/2]*(double)label[ii];
					}
				}
			}
		}
	} 
	else if ((get_kernel()->get_kernel_type()==K_COMBINED) && 
			 (!((CCombinedKernel*)get_kernel())->get_append_subkernel_weights()))// for combined kernel
	{
		CCombinedKernel* k      = (CCombinedKernel*) get_kernel();
		CKernel* kn = k->get_first_kernel() ;
		INT n = 0, i, j ;
		
		while (kn!=NULL)
		{
			for(i=0;i<num;i++) 
			{
				if(a[i] != a_old[i]) 
				{
					kn->get_kernel_row(i,NULL,aicache, true);
					for(j=0;j<num;j++) 
						W[j*num_kernels+n]+=(a[i]-a_old[i])*aicache[regression_fix_index(j)]*(double)label[i];
				}
			}
			kn = k->get_next_kernel(kn) ;
			n++ ;
		}
	}
	else // hope the kernel is fast ...
	{
		DREAL* w_backup = new DREAL[num_kernels] ;
		DREAL* w1 = new DREAL[num_kernels] ;
		
		// backup and set to zero
		for (INT i=0; i<num_kernels; i++)
		{
			w_backup[i] = w[i] ;
			w1[i]=0.0 ; 
		}
		for (INT n=0; n<num_kernels; n++)
		{
			w1[n]=1.0 ;
			k->set_subkernel_weights(w1, num_weights) ;
		
			for(INT i=0;i<num;i++) 
			{
				if(a[i] != a_old[i]) 
				{
					for(INT j=0;j<num;j++) 
						W[j*num_kernels+n]+=(a[i]-a_old[i])*k->kernel(regression_fix_index(i),regression_fix_index(j))*(double)label[i];
				}
			}
			w1[n]=0.0 ;
		}

		// restore old weights
		k->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;
	}
	
	DREAL objective=0;
#ifdef HAVE_LAPACK
	DREAL *alphay  = buffer_num ;
	DREAL sumalpha = 0 ;
	
	for (int i=0; i<num; i++)
	{
		alphay[i]=a[i]*label[i] ;
		sumalpha+=a[i]*(learn_parm->eps-label[i]*c[i]);
	}

	for (int i=0; i<num_kernels; i++)
		sumw[i]=sumalpha ;
	
	cblas_dgemv(CblasColMajor, CblasNoTrans, num_kernels, num,
				0.5, W, num_kernels, alphay, 1, 1.0, sumw, 1) ;
	
	for (int i=0; i<num_kernels; i++)
		objective+=w[i]*sumw[i] ;
#else
	for (int d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for(int i=0; i<num; i++)
			sumw[d] += a[i]*(learn_parm->eps + label[i]*(0.5*W[i*num_kernels+d]-c[i]));
		objective   += w[d]*sumw[d];
	}
#endif
	
	count++ ;
#ifdef USE_CPLEX			
	w_gap = CMath::abs(1-rho/objective) ;
	
	if ((w_gap >= 0.9999*get_weight_epsilon()))
	{
		if (!lp_initialized)
		{
			CIO::message(M_INFO, "creating LP\n") ;
			
			INT NUMCOLS = 2*num_kernels + 1 ;
			double   obj[NUMCOLS];
			double   lb[NUMCOLS];
			double   ub[NUMCOLS];
			for (INT i=0; i<2*num_kernels; i++)
			{
				obj[i]=0 ;
				lb[i]=0 ;
				ub[i]=1 ;
			}
			for (INT i=num_kernels; i<2*num_kernels; i++)
			{
				obj[i]= C_mkl ;
			}
			obj[2*num_kernels]=1 ;
			lb[2*num_kernels]=-CPX_INFBOUND ;
			ub[2*num_kernels]=CPX_INFBOUND ;
			
			INT status = CPXnewcols (env, lp, NUMCOLS, obj, lb, ub, NULL, NULL);
			if ( status ) {
				char  errmsg[1024];
				CPXgeterrorstring (env, status, errmsg);
				CIO::message(M_ERROR, "%s", errmsg);
			}
			
			// add constraint sum(w)=1 ;
			CIO::message(M_INFO, "adding the first row\n") ;
			int rmatbeg[1] ;
			int rmatind[num_kernels+1] ;
			double rmatval[num_kernels+1] ;
			double rhs[1] ;
			char sense[1] ;
			
			rmatbeg[0] = 0;
			rhs[0]=1 ;     // rhs=1 ;
			sense[0]='E' ; // equality
			
			for (INT i=0; i<num_kernels; i++)
			{
				rmatind[i]=i ;
				rmatval[i]=1 ;
			}
			rmatind[num_kernels]=2*num_kernels ;
			rmatval[num_kernels]=0 ;
			
			status = CPXaddrows (env, lp, 0, 1, num_kernels+1, 
								 rhs, sense, rmatbeg,
								 rmatind, rmatval, NULL, NULL);
			if ( status ) {
				CIO::message(M_ERROR, "Failed to add the first row.\n");
			}
			lp_initialized = true ;
			
			if (C_mkl!=0.0)
			{
				for (INT q=0; q<num_kernels-1; q++)
				{
					//fprintf(stderr,"q=%i\n", q) ;
					// add constraint w[i]-w[i+1]<s[i] ;
					// add constraint w[i+1]-w[i]<s[i] ;
					int rmatbeg[1] ;
					int rmatind[3] ;
					double rmatval[3] ;
					double rhs[1] ;
					char sense[1] ;
					
					rmatbeg[0] = 0;
					rhs[0]=0 ;     // rhs=1 ;
					sense[0]='L' ; // equality
					rmatind[0]=q ;
					rmatval[0]=1 ;
					rmatind[1]=q+1 ;
					rmatval[1]=-1 ;
					rmatind[2]=num_kernels+q ;
					rmatval[2]=-1 ;
					status = CPXaddrows (env, lp, 0, 1, 3, 
										 rhs, sense, rmatbeg,
										 rmatind, rmatval, NULL, NULL);
					if ( status ) {
						CIO::message(M_ERROR, "Failed to add a smothness row (1).\n");
					}
					
					rmatbeg[0] = 0;
					rhs[0]=0 ;     // rhs=1 ;
					sense[0]='L' ; // equality
					rmatind[0]=q ;
					rmatval[0]=-1 ;
					rmatind[1]=q+1 ;
					rmatval[1]=1 ;
					rmatind[2]=num_kernels+q ;
					rmatval[2]=-1 ;
					status = CPXaddrows (env, lp, 0, 1, 3, 
										 rhs, sense, rmatbeg,
										 rmatind, rmatval, NULL, NULL);
					if ( status ) {
						CIO::message(M_ERROR, "Failed to add a smothness row (2).\n");
					}
				}
			}
		}

		CIO::message(M_DEBUG, "*") ;
		
		{ // add the new row
			//CIO::message(M_INFO, "add the new row\n") ;
			
			int rmatbeg[1] ;
			int rmatind[num_kernels+1] ;
			double rmatval[num_kernels+1] ;
			double rhs[1] ;
			char sense[1] ;
			
			rmatbeg[0] = 0;
			rhs[0]=0 ;
			sense[0]='L' ;
			
			for (INT i=0; i<num_kernels; i++)
			{
				rmatind[i]=i ;
				rmatval[i]=-sumw[i] ;
			}
			rmatind[num_kernels]=2*num_kernels ;
			rmatval[num_kernels]=-1 ;
			
			INT status = CPXaddrows (env, lp, 0, 1, num_kernels+1, 
									 rhs, sense, rmatbeg,
									 rmatind, rmatval, NULL, NULL);
			if ( status ) 
				CIO::message(M_ERROR, "Failed to add the new row.\n");
		}
		
		{ // optimize
			INT status = CPXlpopt (env, lp);
			if ( status ) 
				CIO::message(M_ERROR, "Failed to optimize LP.\n");
			
			// obtain solution
			INT cur_numrows = CPXgetnumrows (env, lp);
			INT cur_numcols = CPXgetnumcols (env, lp);
			num_rows = cur_numrows ;
			
			if (!buffer_numcols)
				buffer_numcols  = new DREAL[cur_numcols] ;
					
			DREAL *x     = buffer_numcols ;
			DREAL *slack = new DREAL[cur_numrows] ;
			DREAL *pi    = new DREAL[cur_numrows] ;
			
			if ( x     == NULL ||
				 slack == NULL ||
				 pi    == NULL   ) {
				status = CPXERR_NO_MEMORY;
				CIO::message(M_ERROR, "Could not allocate memory for solution.\n") ;
			}
			INT solstat = 0 ;
			DREAL objval = 0 ;
			status = CPXsolution (env, lp, &solstat, &objval, x, pi, slack, NULL);
			INT solution_ok = (!status) ;
			if ( status ) {
				CIO::message(M_ERROR, "Failed to obtain solution.\n");
			}
			
			num_active_rows=0 ;
			if (solution_ok)
			{
				DREAL max_slack = -CMath::INFTY ;
				INT max_idx = -1 ;
				INT start_row = 1 ;
				if (C_mkl!=0.0)
					start_row+=2*(num_kernels-1);

				for (INT i = start_row; i < cur_numrows; i++)  // skip first
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
				
				// have at most max(100,num_active_rows*2) rows, if not, remove one
				if ( (num_rows-start_row>CMath::max(100,2*num_active_rows)) && (max_idx!=-1))
				{
					//CIO::message(M_INFO, "-%i(%i,%i)",max_idx,start_row,num_rows) ;
					INT status = CPXdelrows (env, lp, max_idx, max_idx) ;
					if ( status ) 
						CIO::message(M_ERROR, "Failed to remove an old row.\n");
				}

				// set weights, store new rho and compute new w gap
				k->set_subkernel_weights(x, num_kernels) ;
				rho = -x[2*num_kernels] ;
				w_gap = CMath::abs(1-rho/objective) ;
				
				delete[] pi ;
				delete[] slack ;
			} else
				w_gap = 0 ; // then something is wrong and we rather 
				            // stop sooner than later
		}
	}
#endif
	
	const DREAL* w_new   = k->get_subkernel_weights(num_weights);
	// update lin
#ifdef HAVE_LAPACK
	cblas_dgemv(CblasColMajor,
				CblasTrans, num_kernels, num,
				1.0, W, num_kernels, w_new, 1, 0.0, lin,1);
#else
	for(int i=0; i<num; i++)
		lin[i]=0 ;
	for (int d=0; d<num_kernels; d++)
		if (w_new[d]!=0)
			for(int i=0; i<num; i++)
				lin[i] += w_new[d]*W[i*num_kernels+d] ;
#endif
	
	// count actives
	INT jj ;
	for(jj=0;active2dnum[jj]>=0;jj++);
	
	if (count%10==0)
	{
		INT start_row = 1 ;
		if (C_mkl!=0.0)
			start_row+=2*(num_kernels-1);
		CIO::message(M_DEBUG,"\n%i. OBJ: %f  RHO: %f  wgap=%f agap=%f (activeset=%i; active rows=%i/%i)\n", count, objective,rho,w_gap,mymaxdiff,jj,num_active_rows,num_rows-start_row);
	}
	
	delete[] sumw;
}


void CSVRLight::update_linear_component_mkl_linadd(LONG* docs, INT* label, 
												   long int *active2dnum, double *a, 
												   double *a_old, long int *working2dnum, 
												   long int totdoc,
												   double *lin, DREAL *aicache, double* c)
{
	// kernel with LP_LINADD property is assumed to have 
	// compute_by_subkernel functions
	CKernel* k      = get_kernel();
	INT num         = totdoc;
	INT num_weights = -1;
	INT num_kernels = k->get_num_subkernels() ;
	const DREAL* w   = k->get_subkernel_weights(num_weights);
	
	ASSERT(num_weights==num_kernels) ;
	DREAL* sumw = new DREAL[num_kernels];
	{
		DREAL* w_backup = new DREAL[num_kernels] ;
		DREAL* w1 = new DREAL[num_kernels] ;

		// backup and set to one
		for (INT i=0; i<num_kernels; i++)
		{
			w_backup[i] = w[i] ;
			w1[i]=1.0 ; 
		}
		// set the kernel weights
		k->set_subkernel_weights(w1, num_weights) ;
		
		// create normal update (with changed alphas only)
		k->clear_normal();
		for(INT ii=0, i=0;(i=working2dnum[ii])>=0;ii++) {
			if(a[i] != a_old[i]) {
				k->add_to_normal(regression_fix_index(docs[i]), (a[i]-a_old[i])*(double)label[i]);
			}
		}
		
		// determine contributions of different kernels
		for (int i=0; i<num; i++)
			k->compute_by_subkernel(i,&W[i*num_kernels]) ;

		// restore old weights
		k->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;
	}
	DREAL objective=0;
#ifdef HAVE_LAPACK
	DREAL sumalpha = 0 ;
	
	for (int i=0; i<num; i++)
		sumalpha+=a[i]*(learn_parm->eps-label[i]*c[i]);
	
	for (int i=0; i<num_kernels; i++)
		sumw[i]=-sumalpha ;
	
	cblas_dgemv(CblasColMajor, CblasNoTrans, num_kernels, num,
				0.5, W, num_kernels, a, 1, 1.0, sumw, 1) ;
	
	for (int i=0; i<num_kernels; i++)
		objective+=w[i]*sumw[i] ;
#else
	for (int d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for(int i=0; i<num; i++)
			sumw[d] += a[i]*(learn_parm->eps + label[i]*(0.5*W[i*num_kernels+d]-c[i]));
		objective   += w[d]*sumw[d];
	}
#endif
	
	count++ ;
#ifdef USE_CPLEX			
	w_gap = CMath::abs(1-rho/objective) ;

	if ((w_gap >= 0.9999*get_weight_epsilon()))// && (mymaxdiff < prev_mymaxdiff/2.0))
	{
		CIO::message(M_DEBUG, "*") ;
		if (!lp_initialized)
		{
			CIO::message(M_INFO, "creating LP\n") ;
			
			INT NUMCOLS = 2*num_kernels + 1 ;
			double   obj[NUMCOLS];
			double   lb[NUMCOLS];
			double   ub[NUMCOLS];
			for (INT i=0; i<2*num_kernels; i++)
			{
				obj[i]=0 ;
				lb[i]=0 ;
				ub[i]=1 ;
			}
			for (INT i=num_kernels; i<2*num_kernels; i++)
			{
				obj[i]= C_mkl ;
			}
			obj[2*num_kernels]=1 ;
			lb[2*num_kernels]=-CPX_INFBOUND ;
			ub[2*num_kernels]=CPX_INFBOUND ;
			
			INT status = CPXnewcols (env, lp, NUMCOLS, obj, lb, ub, NULL, NULL);
			if ( status ) {
				char  errmsg[1024];
				CPXgeterrorstring (env, status, errmsg);
				CIO::message(M_ERROR, "%s", errmsg);
			}
			
			// add constraint sum(w)=1 ;
			CIO::message(M_INFO, "add the first row\n") ;
			int rmatbeg[1] ;
			int rmatind[num_kernels+1] ;
			double rmatval[num_kernels+1] ;
			double rhs[1] ;
			char sense[1] ;
			
			rmatbeg[0] = 0;
			rhs[0]=1 ;     // rhs=1 ;
			sense[0]='E' ; // equality
			
			for (INT i=0; i<num_kernels; i++)
			{
				rmatind[i]=i ;
				rmatval[i]=1 ;
			}
			rmatind[num_kernels]=2*num_kernels ;
			rmatval[num_kernels]=0 ;
			
			status = CPXaddrows (env, lp, 0, 1, num_kernels+1, 
								 rhs, sense, rmatbeg,
								 rmatind, rmatval, NULL, NULL);
			if ( status ) {
				CIO::message(M_ERROR, "Failed to add the first row.\n");
			}
			lp_initialized=true ;
			if (C_mkl!=0.0)
			{
				for (INT q=0; q<num_kernels-1; q++)
				{
					// add constraint w[i]-w[i+1]<s[i] ;
					// add constraint w[i+1]-w[i]<s[i] ;
					int rmatbeg[1] ;
					int rmatind[3] ;
					double rmatval[3] ;
					double rhs[1] ;
					char sense[1] ;
					
					rmatbeg[0] = 0;
					rhs[0]=0 ;     // rhs=1 ;
					sense[0]='L' ; // equality
					rmatind[0]=q ;
					rmatval[0]=1 ;
					rmatind[1]=q+1 ;
					rmatval[1]=-1 ;
					rmatind[2]=num_kernels+q ;
					rmatval[2]=-1 ;
					status = CPXaddrows (env, lp, 0, 1, 3, 
										 rhs, sense, rmatbeg,
										 rmatind, rmatval, NULL, NULL);
					if ( status ) {
						CIO::message(M_ERROR, "Failed to add a smothness row (1).\n");
					}
					
					rmatbeg[0] = 0;
					rhs[0]=0 ;     // rhs=1 ;
					sense[0]='L' ; // equality
					rmatind[0]=q ;
					rmatval[0]=-1 ;
					rmatind[1]=q+1 ;
					rmatval[1]=1 ;
					rmatind[2]=num_kernels+q ;
					rmatval[2]=-1 ;
					status = CPXaddrows (env, lp, 0, 1, 3, 
										 rhs, sense, rmatbeg,
										 rmatind, rmatval, NULL, NULL);
					if ( status ) {
						CIO::message(M_ERROR, "Failed to add a smothness row (2).\n");
					}
				}
			}
		}
		
		{ // add the new row
			//CIO::message(M_INFO, "add the new row\n") ;
			
			int rmatbeg[1] ;
			int rmatind[num_kernels+1] ;
			double rmatval[num_kernels+1] ;
			double rhs[1] ;
			char sense[1] ;
			
			rmatbeg[0] = 0;
			rhs[0]=0 ;
			sense[0]='L' ;
			
			for (INT i=0; i<num_kernels; i++)
			{
				rmatind[i]=i ;
				rmatval[i]=-sumw[i] ;
			}
			rmatind[num_kernels]=2*num_kernels ;
			rmatval[num_kernels]=-1 ;
			
			INT status = CPXaddrows (env, lp, 0, 1, num_kernels+1, 
									 rhs, sense, rmatbeg,
									 rmatind, rmatval, NULL, NULL);
			if ( status ) 
				CIO::message(M_ERROR, "Failed to add the new row.\n");
		}
		
		{ // optimize
			INT status = CPXlpopt (env, lp);
			if ( status ) 
				CIO::message(M_ERROR, "Failed to optimize LP.\n");
			
			// obtain solution
			INT cur_numrows = CPXgetnumrows (env, lp);
			INT cur_numcols = CPXgetnumcols (env, lp);
			num_rows = cur_numrows ;
			
			if (!buffer_numcols)
				buffer_numcols  = new DREAL[cur_numcols] ;
					
			DREAL *x     = buffer_numcols ;
			DREAL *slack = new DREAL[cur_numrows] ;
			DREAL *pi    = new DREAL[cur_numrows] ;
			
			if ( x     == NULL ||
				 slack == NULL ||
				 pi    == NULL   ) {
				status = CPXERR_NO_MEMORY;
				CIO::message(M_ERROR, "Could not allocate memory for solution.\n") ;
			}
			INT solstat = 0 ;
			DREAL objval = 0 ;
			status = CPXsolution (env, lp, &solstat, &objval, x, pi, slack, NULL);
			INT solution_ok = (!status) ;
			if ( status ) {
				CIO::message(M_ERROR, "Failed to obtain solution.\n");
			}
			
			num_active_rows=0 ;
			if (solution_ok)
			{
				DREAL max_slack = -CMath::INFTY ;
				INT max_idx = -1 ;
				INT start_row = 1 ;
				if (C_mkl!=0.0)
					start_row+=2*(num_kernels-1);

				for (INT i = start_row; i < cur_numrows; i++)  // skip first
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
				
				// have at most max(100,num_active_rows*2) rows, if not, remove one
				if ( (num_rows-start_row>CMath::max(100,2*num_active_rows)) && (max_idx!=-1))
				{
					//CIO::message(M_INFO, "-%i(%i,%i)",max_idx,start_row,num_rows) ;
					INT status = CPXdelrows (env, lp, max_idx, max_idx) ;
					if ( status ) 
						CIO::message(M_ERROR, "Failed to remove an old row.\n");
				}

				// set weights, store new rho and compute new w gap
				k->set_subkernel_weights(x, num_kernels) ;
				rho = -x[2*num_kernels] ;
				w_gap = CMath::abs(1-rho/objective) ;
				
				delete[] pi ;
				delete[] slack ;
			} else
				w_gap = 0 ; // then something is wrong and we rather 
				            // stop sooner than later
		}
	}
#endif
	
	// update lin
#ifdef HAVE_LAPACK
	cblas_dgemv(CblasColMajor,
				CblasTrans, num_kernels, num,
				1.0, W, num_kernels, w, 1, 0.0, lin,1);
#else
	for(int i=0; i<num; i++)
		lin[i]=0 ;
	for (int d=0; d<num_kernels; d++)
		if (w[d]!=0)
			for(int i=0; i<num; i++)
				lin[i] += w[d]*W[i*num_kernels+d] ;
#endif
	
	// count actives
	INT jj ;
	for(jj=0;active2dnum[jj]>=0;jj++);
	
	if (count%10==0)
	{
		INT start_row = 1 ;
		if (C_mkl!=0.0)
			start_row+=2*(num_kernels-1);
		CIO::message(M_DEBUG,"\n%i. OBJ: %f  RHO: %f  wgap=%f agap=%f (activeset=%i; active rows=%i/%i)\n", count, objective,rho,w_gap,mymaxdiff,jj,num_active_rows,num_rows-start_row);
	}
	
	delete[] sumw;

}


void* CSVRLight::update_linear_component_linadd_helper(void *params_)
{
	S_THREAD_PARAM * params = (S_THREAD_PARAM*) params_ ;
	
	INT jj=0, j=0 ;
	
	for(jj=params->start;(jj<params->end) && (j=params->active2dnum[jj])>=0;jj++) 
		params->lin[j]+=params->kernel->compute_optimized(CSVRLight::regression_fix_index2(params->docs[j], params->num_vectors));

	return NULL ;
}


void CSVRLight::update_linear_component(LONG* docs, INT* label, 
										long int *active2dnum, double *a, 
										double *a_old, long int *working2dnum, 
										long int totdoc,
										double *lin, DREAL *aicache, double* c)
     /* keep track of the linear component */
     /* lin of the gradient etc. by updating */
     /* based on the change of the variables */
     /* in the current working set */
{
	register long i=0,ii=0,j=0,jj=0;

	if (get_kernel()->has_property(KP_LINADD) && get_linadd_enabled()) 
	{
		if (get_kernel()->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() ) 
		{
			update_linear_component_mkl_linadd(docs, label, active2dnum, a, a_old, working2dnum, 
											   totdoc,	lin, aicache, c) ;
		}
		else
		{
			get_kernel()->clear_normal();

			INT num_working=0;
			for(ii=0;(i=working2dnum[ii])>=0;ii++) {
				if(a[i] != a_old[i]) {
					get_kernel()->add_to_normal(regression_fix_index(docs[i]), (a[i]-a_old[i])*(double)label[i]);
					num_working++;
				}
			}

			if (num_working>0)
			{
				if (CParallel::get_num_threads() < 2)
				{
					for(jj=0;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=get_kernel()->compute_optimized(regression_fix_index(docs[j]));
					}
				}
				else
				{
					INT num_elem = 0 ;
					for(jj=0;(j=active2dnum[jj])>=0;jj++) num_elem++ ;

					pthread_t threads[CParallel::get_num_threads()-1] ;
					S_THREAD_PARAM params[CParallel::get_num_threads()-1] ;
					INT start = 0 ;
					INT step = num_elem/CParallel::get_num_threads() ;
					INT end = step ;

					for (INT t=0; t<CParallel::get_num_threads()-1; t++)
					{
						params[t].kernel = get_kernel() ;
						params[t].lin = lin ;
						params[t].docs = docs ;
						params[t].active2dnum=active2dnum ;
						params[t].start = start ;
						params[t].end = end ;
						params[t].num_vectors=num_vectors ;

						start=end ;
						end+=step ;
						pthread_create(&threads[t], NULL, update_linear_component_linadd_helper, (void*)&params[t]) ;
					}

					for(jj=params[CParallel::get_num_threads()-2].end;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=get_kernel()->compute_optimized(regression_fix_index(docs[j]));
					}
					void* ret;
					for (INT t=0; t<CParallel::get_num_threads()-1; t++)
						pthread_join(threads[t], &ret) ;
				}
			}
		}
	}
	else 
	{
		if (get_kernel()->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() ) 
		{
			update_linear_component_mkl(docs, label, active2dnum, a, a_old, working2dnum, 
										totdoc,	lin, aicache, c) ;
		}
		else {
			for(jj=0;(i=working2dnum[jj])>=0;jj++) {
				if(a[i] != a_old[i]) {
					CKernelMachine::get_kernel()->get_kernel_row(i,active2dnum,aicache);
					for(ii=0;(j=active2dnum[ii])>=0;ii++)
						lin[j]+=(a[i]-a_old[i])*aicache[j]*(double)label[i];
				}
			}
		}
	}
}


void CSVRLight::reactivate_inactive_examples(INT* label, 
				  double *a, 
				  SHRINK_STATE *shrink_state, 
				  double *lin, 
				  double *c, 
				  long int totdoc, 
				  long int iteration, 
				  long int *inconsistent, 
				  LONG* docs, 
				  MODEL *model, 
				  DREAL *aicache, 
				  double *maxdiff)
     /* Make all variables active again which had been removed by
        shrinking. */
     /* Computes lin for those variables from scratch. */
{
  register long i=0,j,ii=0,jj,t,*changed2dnum,*inactive2dnum;
  long *changed,*inactive;
  register double *a_old,dist;
  double ex_c,target;

  if (get_kernel()->has_property(KP_LINADD) && get_linadd_enabled()) { /* special linear case */
	  a_old=shrink_state->last_a;    

	  get_kernel()->clear_normal();
	  INT num_modified=0;
	  for(i=0;i<totdoc;i++) {
		  if(a[i] != a_old[i]) {
			  get_kernel()->add_to_normal(regression_fix_index(docs[i]), ((a[i]-a_old[i])*(double)label[i]));
			  a_old[i]=a[i];
			  num_modified++;
		  }
	  }

	  if (num_modified>0)
	  {
		  for(i=0;i<totdoc;i++) {
			  if(!shrink_state->active[i]) {
				  lin[i]=shrink_state->last_lin[i]+get_kernel()->compute_optimized(regression_fix_index(docs[i]));
			  }
			  shrink_state->last_lin[i]=lin[i];
		  }
	  }
  }
  else 
  {
	  changed=new long[totdoc];
	  changed2dnum=new long[totdoc+11];
	  inactive=new long[totdoc];
	  inactive2dnum=new long[totdoc+11];
	  for(t=shrink_state->deactnum-1;(t>=0) && shrink_state->a_history[t];t--) {
		  if(verbosity>=2) {
			  CIO::message(M_INFO, "%ld..",t);
		  }
		  a_old=shrink_state->a_history[t];    
		  for(i=0;i<totdoc;i++) {
			  inactive[i]=((!shrink_state->active[i]) 
					  && (shrink_state->inactive_since[i] == t));
			  changed[i]= (a[i] != a_old[i]);
		  }
		  compute_index(inactive,totdoc,inactive2dnum);
		  compute_index(changed,totdoc,changed2dnum);

		  for(ii=0;(i=changed2dnum[ii])>=0;ii++) {
			  CKernelMachine::get_kernel()->get_kernel_row(i,inactive2dnum,aicache);
			  for(jj=0;(j=inactive2dnum[jj])>=0;jj++)
				  lin[j]+=(a[i]-a_old[i])*aicache[j]*(double)label[i];
		  }
	  }
	  delete[] changed;
	  delete[] changed2dnum;
	  delete[] inactive;
	  delete[] inactive2dnum;
  }

  (*maxdiff)=0;
  for(i=0;i<totdoc;i++) {
    shrink_state->inactive_since[i]=shrink_state->deactnum-1;
    if(!inconsistent[i]) {
      dist=(lin[i]-model->b)*(double)label[i];
      target=-(learn_parm->eps-(double)label[i]*c[i]);
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if((a[i]>learn_parm->epsilon_a) && (dist > target)) {
	if((dist-target)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-target;
      }
      else if((a[i]<ex_c) && (dist < target)) {
	if((target-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=target-dist;
      }
      if((a[i]>(0+learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	shrink_state->active[i]=1;                         /* not at bound */
      }
      else if((a[i]<=(0+learn_parm->epsilon_a)) && (dist < (target+learn_parm->epsilon_shrink))) {
	shrink_state->active[i]=1;
      }
      else if((a[i]>=ex_c)
	      && (dist > (target-learn_parm->epsilon_shrink))) {
	shrink_state->active[i]=1;
      }
      else if(learn_parm->sharedslack) { /* make all active when sharedslack */
	shrink_state->active[i]=1;
      }
    }
  }
  if (use_kernel_cache) { /* update history for non-linear */
	  for(i=0;i<totdoc;i++) {
		  (shrink_state->a_history[shrink_state->deactnum-1])[i]=a[i];
	  }
	  for(t=shrink_state->deactnum-2;(t>=0) && shrink_state->a_history[t];t--) {
		  delete[] shrink_state->a_history[t];
		  shrink_state->a_history[t]=0;
	  }
  }
}
#endif //USE_SVMLIGHT

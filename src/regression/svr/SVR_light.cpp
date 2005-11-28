#include "lib/config.h"

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

#include <assert.h>
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
	REAL * lin ;
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
	opt_precision=DEF_PRECISION_LINEAR;

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
	opt_precision=DEF_PRECISION_LINEAR;
	
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

	// MKL stuff
	buffer_num = new REAL[get_kernel()->get_rhs()->get_num_vectors()] ;
	delete[] buffer_numcols ;
	buffer_numcols = NULL ;

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
						 (get_mkl_enabled() && get_kernel()->has_property(KP_KERNCOMBINATION))||
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
		const REAL* w   = k->get_subkernel_weights(num_weights);

		REAL* w_backup = new REAL[num_kernels] ;
		REAL* w1 = new REAL[num_kernels] ;
		
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
			assert(precomputed_subkernels[n]!=NULL) ;
		}
		
		for (INT n=0; n<num_precomputed_subkernels; n++)
		{
			w1[n]=1.0 ;
			k->set_subkernel_weights(w1, num_weights) ;

			SHORTREAL * matrix = precomputed_subkernels[n] ;
			
			CIO::message(M_INFO, "precomputing kernel matrix %i (%ix%i)\n", n, num, num) ;
			for (INT i=0; i<num; i++)
			{
				// CIO::message(M_INFO, "\r %1.2f%% ", 100.0*i*i/(num*num)) ;
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
	assert(lab!=NULL);
	INT totdoc=lab->get_num_labels();
	num_vectors=totdoc;
	
	/* set up regression problem in standard form */
	docs=new long[2*totdoc];
	label=new INT[2*totdoc];
	c = new double[2*totdoc];

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

  //prepare kernel cache for regression (i.e. cachelines are twice of current size)
  get_kernel()->resize_kernel_cache( get_kernel()->get_cache_size(), true);

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
		W = new REAL[totdoc*get_kernel()->get_num_subkernels()];
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
			assert(false);
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
		CIO::message(M_INFO, "Optimization finished (maxdiff=%.5f).\n",maxdiff);
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
}

long CSVRLight::optimize_to_convergence(LONG* docs, INT* label, long int totdoc, 
			     SHRINK_STATE *shrink_state, MODEL *model, 
			     long int *inconsistent,
			     double *a, double *lin, double *c, 
			     TIMING *timing_profile, double *maxdiff, 
			     long int heldout, long int retrain)
     /* docs: Training vectors (x-part) */
     /* label: Training labels/value (y-part, zero if test example for
			      transduction) */
     /* totdoc: Number of examples in docs/label */
     /* laern_parm: Learning paramenters */
     /* kernel_parm: Kernel paramenters */
     /* kernel_cache: Initialized/partly filled Cache, if using a kernel. 
                      NULL if linear. */
     /* shrink_state: State of active variables */
     /* model: Returns learning result */
     /* inconsistent: examples thrown out as inconstistent */
     /* a: alphas */
     /* lin: linear component of gradient */
     /* c: right hand side of inequalities (margin) */
     /* maxdiff: returns maximum violation of KT-conditions */
     /* heldout: marks held-out example for leave-one-out (or -1) */
     /* retrain: selects training mode (1=regular / 2=holdout) */
{
  long *chosen,*key,i,j,jj,*last_suboptimal_at,noshrink;
  long inconsistentnum,choosenum,already_chosen=0,iteration;
  long misclassified,supvecnum=0,*active2dnum,inactivenum;
  long *working2dnum,*selexam;
  long activenum;
  double criterion, eq;
  double *a_old;
  long t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  long transductcycle;
  long transduction;
  double epsilon_crit_org; 
  double bestmaxdiff;
  double worstmaxdiff;
  long   bestmaxdiffiter,terminate;

  double *selcrit;  /* buffer for sorting */        
  REAL *aicache;  /* buffer to keep one row of hessian */
  QP qp;            /* buffer for one quadratic program */

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
  if(get_kernel()->has_property(KP_LINADD) && get_linadd_enabled()) {
	  learn_parm->epsilon_crit=2.0;
      /* caching makes no sense for linear kernel */
  } 
  learn_parm->epsilon_shrink=2;
  (*maxdiff)=1;

  chosen = new long[totdoc];
  last_suboptimal_at =new long[totdoc];
  key =new long[totdoc+11];
  selcrit =new double[totdoc];
  selexam =new long[totdoc];
  a_old =new double[totdoc];
  aicache =new REAL[totdoc];
  working2dnum =new long[totdoc+11];
  active2dnum =new long[totdoc+11];
  qp.opt_ce =new double[learn_parm->svm_maxqpsize];
  qp.opt_ce0 =new double[1];
  qp.opt_g =new double[learn_parm->svm_maxqpsize*learn_parm->svm_maxqpsize];
  qp.opt_g0 =new double[learn_parm->svm_maxqpsize];
  qp.opt_xinit =new double[learn_parm->svm_maxqpsize];
  qp.opt_low=new double[learn_parm->svm_maxqpsize];
  qp.opt_up=new double[learn_parm->svm_maxqpsize];

  choosenum=0;
  inconsistentnum=0;
  transductcycle=0;
  transduction=0;
  if(!retrain) retrain=1;
  iteration=1;
  bestmaxdiffiter=1;
  bestmaxdiff=999999999;
  worstmaxdiff=1e-10;
  terminate=0;
  
  CKernelMachine::get_kernel()->set_time(iteration);  /* for lru cache */
  if (use_kernel_cache)
	  CKernelMachine::get_kernel()->kernel_cache_reset_lru();


  for(i=0;i<totdoc;i++) {    /* various inits */
    chosen[i]=0;
    a_old[i]=a[i];
    last_suboptimal_at[i]=1;
    if(inconsistent[i]) 
      inconsistentnum++;
  }
  activenum=compute_index(shrink_state->active,totdoc,active2dnum);
  inactivenum=totdoc-activenum;
  clear_index(working2dnum);

                            /* repeat this loop until we have convergence */

  for(;((!CSignal::cancel_computations()) && ((iteration<3) || (retrain && (!terminate))||((w_gap>get_weight_epsilon()) && get_mkl_enabled()))); iteration++){

#ifdef HAVE_PYTHON
	  if (PyErr_CheckSignals())
		  break;
#endif
	  	  
	  if(use_kernel_cache) 
		  CKernelMachine::get_kernel()->set_time(iteration);  /* for lru cache */
	  
	  if(verbosity>=2) t0=get_runtime();
	  if(verbosity>=3) {
		  CIO::message(M_DEBUG, "\nSelecting working set...%f "); 
	  }
	  
	  if(learn_parm->svm_newvarsinqp>learn_parm->svm_maxqpsize) 
		  learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;
	  
	  i=0;
	  for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear working set */
		  if((chosen[j]>=(learn_parm->svm_maxqpsize/
						  CMath::min(learn_parm->svm_maxqpsize,
									 learn_parm->svm_newvarsinqp))) 
			 || (inconsistent[j])
			 || (j == heldout)) {
			  chosen[j]=0; 
			  choosenum--; 
		  }
		  else {
			  chosen[j]++;
			  working2dnum[i++]=j;
		  }
	  }
	  working2dnum[i]=-1;
	  
	  if(retrain == 2) {
		  choosenum=0;
		  for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* fully clear working set */
			  chosen[j]=0; 
		  }
		  clear_index(working2dnum);
		  for(i=0;i<totdoc;i++) { /* set inconsistent examples to zero (-i 1) */
			  if((inconsistent[i] || (heldout==i)) && (a[i] != 0.0)) {
				  chosen[i]=99999;
				  choosenum++;
				  a[i]=0;
			  }
		  }
		  if(learn_parm->biased_hyperplane) {
			  eq=0;
			  for(i=0;i<totdoc;i++) { /* make sure we fulfill equality constraint */
				  eq+=a[i]*label[i];
			  }
			  for(i=0;(i<totdoc) && (fabs(eq) > learn_parm->epsilon_a);i++) {
				  if((eq*label[i] > 0) && (a[i] > 0)) {
					  chosen[i]=88888;
					  choosenum++;
					  if((eq*label[i]) > a[i]) {
						  eq-=(a[i]*label[i]);
						  a[i]=0;
					  }
					  else {
						  a[i]-=(eq*label[i]);
						  eq=0;
					  }
				  }
			  }
		  }
		  compute_index(chosen,totdoc,working2dnum);
	  }
	  else
	  {   /* select working set according to steepest gradient */
		  if(iteration % 101)
		  {
			  already_chosen=0;
			  if(CMath::min(learn_parm->svm_newvarsinqp, learn_parm->svm_maxqpsize-choosenum)>=4 && use_kernel_cache)
			  {
				  /* select part of the working set from cache */
				  already_chosen=select_next_qp_subproblem_grad(
					  label,a,lin,c,totdoc,
					  (long)(CMath::min(learn_parm->svm_maxqpsize-choosenum,
										learn_parm->svm_newvarsinqp)/2),
					  inconsistent,active2dnum,
					  working2dnum,selcrit,selexam,1,
					  key,chosen);
				  choosenum+=already_chosen;
			  }
			  choosenum+=select_next_qp_subproblem_grad(
				  label,a,lin,c,totdoc,
				  CMath::min(learn_parm->svm_maxqpsize-choosenum,
							 learn_parm->svm_newvarsinqp-already_chosen),
				  inconsistent,active2dnum,
				  working2dnum,selcrit,selexam,0,key,
				  chosen);
		  }
		  else { /* once in a while, select a somewhat random working set
					to get unlocked of infinite loops due to numerical
					inaccuracies in the core qp-solver */
			  choosenum+=select_next_qp_subproblem_rand(
				  label,a,lin,c,totdoc,
				  CMath::min(learn_parm->svm_maxqpsize-choosenum,
							 learn_parm->svm_newvarsinqp),
				  inconsistent,active2dnum,
				  working2dnum,selcrit,selexam,key,
				  chosen,iteration);
		  }
	  }
	  
	  if(verbosity>=2) {
		  CIO::message(M_INFO, " %ld vectors chosen\n",choosenum); 
	  }
	  
	  if(verbosity>=2) t1=get_runtime();
	  
	  if (use_kernel_cache)
	  {
		  // in case of MKL w/o linadd ALSO cache each kernel independently
		  if ( get_kernel()->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() &&
				  (!get_kernel()->has_property(KP_LINADD) || !get_linadd_enabled()) &&
				  (!((CCombinedKernel*)get_kernel())->get_append_subkernel_weights()) 
			 )
		  {
			  CCombinedKernel* k      = (CCombinedKernel*) get_kernel();
			  CKernel* kn = k->get_first_kernel();

			  while (kn)
			  {
				  kn->cache_multiple_kernel_rows(working2dnum, choosenum); 
				  kn = k->get_next_kernel(kn) ;
			  }
		  }
		  else
			  CKernelMachine::get_kernel()->cache_multiple_kernel_rows(working2dnum, choosenum); 
	  }
	  
	  if(verbosity>=2) t2=get_runtime();
	  if(retrain != 2) {
		  optimize_svm(docs,label,inconsistent,0.0,chosen,active2dnum,
					   model,totdoc,working2dnum,choosenum,a,lin,c,
					   aicache,&qp,&epsilon_crit_org);
	  }
	  
	  if(verbosity>=2) t3=get_runtime();
	  update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
							  lin,aicache,c);
	  
	  if(verbosity>=2) t4=get_runtime();
	  supvecnum=calculate_svm_model(docs,label,lin,a,a_old,c,working2dnum,active2dnum,model);
	  
	  if(verbosity>=2) t5=get_runtime();

	  for(jj=0;(i=working2dnum[jj])>=0;jj++) {
		  a_old[i]=a[i];
	  }
	  
	  retrain=check_optimality(model,label,a,lin,c,totdoc,
							   maxdiff,epsilon_crit_org,&misclassified,
							   inconsistent,active2dnum,last_suboptimal_at,
							   iteration);
	  
	  if(verbosity>=2) {
		  t6=get_runtime();
		  timing_profile->time_select+=t1-t0;
		  timing_profile->time_kernel+=t2-t1;
		  timing_profile->time_opti+=t3-t2;
		  timing_profile->time_update+=t4-t3;
		  timing_profile->time_model+=t5-t4;
		  timing_profile->time_check+=t6-t5;
	  }
	  
	  /* checking whether optimizer got stuck */
	  if((*maxdiff) < bestmaxdiff) {
		  bestmaxdiff=(*maxdiff);
		  bestmaxdiffiter=iteration;
	  }
	  if(iteration > (bestmaxdiffiter+learn_parm->maxiter)) { 
		  /* long time no progress? */
		  terminate=1;
		  retrain=0;
		  CIO::message(M_WARN, "Relaxing KT-Conditions due to slow progress! Terminating!\n");
	  }
	  
	  noshrink=0;
	  if ((!retrain) && (inactivenum>0) && (!learn_parm->skip_final_opt_check)
		  /*|| (get_kernel()->has_property(KP_LINADD))*/) { 
		  t1=get_runtime();
		  reactivate_inactive_examples(label,a,shrink_state,lin,c,totdoc,
									   iteration,inconsistent,
									   docs,model,aicache,
									   maxdiff);
		  /* Update to new active variables. */
		  activenum=compute_index(shrink_state->active,totdoc,active2dnum);
		  inactivenum=totdoc-activenum;
		  /* reset watchdog */
		  bestmaxdiff=(*maxdiff);
		  bestmaxdiffiter=iteration;

		  /* termination criterion */
		  noshrink=1;
		  retrain=0;
		  if((*maxdiff) > learn_parm->epsilon_crit) 
			  retrain=1;
		  timing_profile->time_shrink+=get_runtime()-t1;
		  if (((verbosity>=1) && use_kernel_cache)
			 || (verbosity>=2)) {
			  CIO::message(M_INFO, "done.\n");
			  CIO::message(M_INFO, "Number of inactive variables = %ld\n",inactivenum);
		  }		  
	  }
	  
	  if((!retrain) && (learn_parm->epsilon_crit>(*maxdiff))) 
		  learn_parm->epsilon_crit=(*maxdiff);
	  if((!retrain) && (learn_parm->epsilon_crit>epsilon_crit_org)) {
		  learn_parm->epsilon_crit/=2.0;
		  retrain=1;
		  noshrink=1;
	  }
	  if(learn_parm->epsilon_crit<epsilon_crit_org) 
		  learn_parm->epsilon_crit=epsilon_crit_org;
	  
	  if(verbosity>=2) {
		  CIO::message(M_INFO, " => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
					   supvecnum,model->at_upper_bound,(*maxdiff)); 
		  
	  }
	  mymaxdiff=*maxdiff ;

	  
	  if (((iteration % 10) == 0) && (!noshrink))
	  {
		  activenum=shrink_problem(shrink_state,active2dnum,last_suboptimal_at,iteration,totdoc,
								   CMath::max((LONG)(activenum/10),
											  CMath::max((LONG)(totdoc/500),(LONG) 100)),
								   a,inconsistent);
		  inactivenum=totdoc-activenum;
		  
		  if (use_kernel_cache)
		  {
			  if( (supvecnum>get_kernel()->get_max_elems_cache()) && ((get_kernel()->get_activenum_cache()-activenum)>CMath::max((LONG)(activenum/10),(LONG) 500))) {
				  get_kernel()->kernel_cache_shrink(totdoc, CMath::min((LONG) (get_kernel()->get_activenum_cache()-activenum),
																	   (LONG) (get_kernel()->get_activenum_cache()-supvecnum)),
													shrink_state->active); 
			  }
		  }
	  }

	  if (bestmaxdiff>worstmaxdiff)
		  worstmaxdiff=bestmaxdiff;

	  //CIO::progress(-CMath::log10(bestmaxdiff), -CMath::log10(worstmaxdiff), -CMath::log10(epsilon), 6);
	  CIO::absolute_progress(bestmaxdiff, -CMath::log10(bestmaxdiff), -CMath::log10(worstmaxdiff), -CMath::log10(epsilon), 6);
  } /* end of loop */

  /* The following computation of the objective function works only */
  /* relative to the active variables */
  criterion=compute_objective_function(a,lin,c,learn_parm->eps,label, active2dnum);
  CSVM::set_objective(criterion);
	  
  delete[] chosen;
  delete[] last_suboptimal_at;
  delete[] key;
  delete[] selcrit;
  delete[] selexam;
  delete[] a_old;
  delete[] aicache;
  delete[] working2dnum;
  delete[] active2dnum;
  delete[] qp.opt_ce;
  delete[] qp.opt_ce0;
  delete[] qp.opt_g;
  delete[] qp.opt_g0;
  delete[] qp.opt_xinit;
  delete[] qp.opt_low;
  delete[] qp.opt_up;

  learn_parm->epsilon_crit=epsilon_crit_org; /* restore org */

  return(iteration);
}

double CSVRLight::compute_objective_function(double *a, double *lin, double *c,
                  double eps, INT *label,
                  long int *active2dnum)
     /* Return value of objective function. */
     /* Works only relative to the active variables! */
{
  long i,ii;
  double criterion;
  /* calculate value of objective function */
  criterion=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    criterion=criterion+(eps-(double)label[i]*c[i])*a[i]+0.5*a[i]*label[i]*lin[i];
  }
  return(criterion);
}


void CSVRLight::update_linear_component_mkl(LONG* docs, INT* label, 
											long int *active2dnum, double *a, 
											double *a_old, long int *working2dnum, 
											long int totdoc,
											double *lin, REAL *aicache, double* c)
{
	CKernel* k      = get_kernel();
	INT num         = totdoc;
	INT num_weights = -1;
	INT num_kernels = k->get_num_subkernels() ;
	const REAL* w   = k->get_subkernel_weights(num_weights);

	assert(num_weights==num_kernels) ;
	REAL* sumw = new REAL[num_kernels];

	if (use_precomputed_subkernels) // everything is already precomputed
	{
		assert(precomputed_subkernels!=NULL) ;
		for (INT n=0; n<num_kernels; n++)
		{
			assert(precomputed_subkernels[n]!=NULL) ;
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
							W[j*num_kernels+n]+=(a[ii]-a_old[ii])*matrix[i*(i+1)/2+j]*(double)label[ii];
						else
							W[j*num_kernels+n]+=(a[ii]-a_old[ii])*matrix[i+j*(j+1)/2]*(double)label[ii];
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
					kn->get_kernel_row(i,active2dnum,aicache);
					for(int ii=0;(j=active2dnum[ii])>=0;ii++) {
						W[j*num_kernels+n]+=(((a[i]*aicache[j])-(a_old[i]*aicache[j]))*(double)label[i]);
					}
				}
			}
			kn = k->get_next_kernel(kn) ;
			n++ ;
		}
	}
	else // hope the kernel is fast ...
	{
		REAL* w_backup = new REAL[num_kernels] ;
		REAL* w1 = new REAL[num_kernels] ;
		
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
						W[j*num_kernels+n]+=(a[i]-a_old[i])*k->kernel(i,j)*(double)label[i];
				}
			}
			w1[n]=0.0 ;
		}

		// restore old weights
		k->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;
	}
	
	REAL objective=0;
#ifdef HAVE_LAPACK
	REAL *alphay  = buffer_num ;
	REAL sumalpha = 0 ;
	
#warning untested, verify me
	for (int i=0; i<num; i++)
	{
		alphay[i]=a[i]*label[i] ;
		sumalpha+=a[i]*(learn_parm->eps-label[i]*c[i]);
	}
	
	for (int i=0; i<num_kernels; i++)
		sumw[i]=-sumalpha ;
	
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
	
	//CIO::message(M_INFO, "(%i) ",count) ;
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
				buffer_numcols  = new REAL[cur_numcols] ;
					
			REAL *x     = buffer_numcols ;
			REAL *slack = new REAL[cur_numrows] ;
			REAL *pi    = new REAL[cur_numrows] ;
			
			if ( x     == NULL ||
				 slack == NULL ||
				 pi    == NULL   ) {
				status = CPXERR_NO_MEMORY;
				CIO::message(M_ERROR, "Could not allocate memory for solution.\n") ;
			}
			INT solstat = 0 ;
			REAL objval = 0 ;
			status = CPXsolution (env, lp, &solstat, &objval, x, pi, slack, NULL);
			INT solution_ok = (!status) ;
			if ( status ) {
				CIO::message(M_ERROR, "Failed to obtain solution.\n");
			}
			
			num_active_rows=0 ;
			if (solution_ok)
			{
				REAL max_slack = -CMath::INFTY ;
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


void CSVRLight::update_linear_component_mkl_linadd(LONG* docs, INT* label, 
												   long int *active2dnum, double *a, 
												   double *a_old, long int *working2dnum, 
												   long int totdoc,
												   double *lin, REAL *aicache, double* c)
{
	// kernel with LP_LINADD property is assumed to have 
	// compute_by_subkernel functions
	CKernel* k      = get_kernel();
	INT num         = totdoc;
	INT num_weights = -1;
	INT num_kernels = k->get_num_subkernels() ;
	const REAL* w   = k->get_subkernel_weights(num_weights);
	
	assert(num_weights==num_kernels) ;
	REAL* sumw = new REAL[num_kernels];
	{
		REAL* w_backup = new REAL[num_kernels] ;
		REAL* w1 = new REAL[num_kernels] ;

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
	REAL objective=0;
#ifdef HAVE_LAPACK
	REAL sumalpha = 0 ;
	
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
				buffer_numcols  = new REAL[cur_numcols] ;
					
			REAL *x     = buffer_numcols ;
			REAL *slack = new REAL[cur_numrows] ;
			REAL *pi    = new REAL[cur_numrows] ;
			
			if ( x     == NULL ||
				 slack == NULL ||
				 pi    == NULL   ) {
				status = CPXERR_NO_MEMORY;
				CIO::message(M_ERROR, "Could not allocate memory for solution.\n") ;
			}
			INT solstat = 0 ;
			REAL objval = 0 ;
			status = CPXsolution (env, lp, &solstat, &objval, x, pi, slack, NULL);
			INT solution_ok = (!status) ;
			if ( status ) {
				CIO::message(M_ERROR, "Failed to obtain solution.\n");
			}
			
			num_active_rows=0 ;
			if (solution_ok)
			{
				REAL max_slack = -CMath::INFTY ;
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

inline INT regression_fix_index2(INT i,INT num_vectors)
{
	if (i>=num_vectors)
		i=2*num_vectors-1-i;

	return i;
}

void* CSVRLight::update_linear_component_linadd_helper(void *params_)
{
	S_THREAD_PARAM * params = (S_THREAD_PARAM*) params_ ;
	
	INT jj=0, j=0 ;
	
	for(jj=params->start;(jj<params->end) && (j=params->active2dnum[jj])>=0;jj++) 
	{
		params->lin[j]+=params->kernel->compute_optimized(regression_fix_index2(params->docs[j], params->num_vectors));
	}
	return NULL ;
}


void CSVRLight::update_linear_component(LONG* docs, INT* label, 
										long int *active2dnum, double *a, 
										double *a_old, long int *working2dnum, 
										long int totdoc,
										double *lin, REAL *aicache, double* c)
     /* keep track of the linear component */
     /* lin of the gradient etc. by updating */
     /* based on the change of the variables */
     /* in the current working set */
{
	register long i=0,ii=0,j=0,jj=0;
	register double tec=0;


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

			for(ii=0;(i=working2dnum[ii])>=0;ii++) {
				if(a[i] != a_old[i]) {
					get_kernel()->add_to_normal(regression_fix_index(docs[i]), (a[i]-a_old[i])*(double)label[i]);
				}
			}

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
					for(ii=0;(j=active2dnum[ii])>=0;ii++) {
						tec=aicache[j];
						lin[j]+=(((a[i]*tec)-(a_old[i]*tec))*(double)label[i]);
					}
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
				  REAL *aicache, 
				  double *maxdiff)
     /* Make all variables active again which had been removed by
        shrinking. */
     /* Computes lin for those variables from scratch. */
{
  register long i=0,j,ii=0,jj,t,*changed2dnum,*inactive2dnum;
  long *changed,*inactive;
  register double kernel_val,*a_old,dist;
  double ex_c,target;

  if (get_kernel()->has_property(KP_LINADD) && get_linadd_enabled()) { /* special linear case */
	  a_old=shrink_state->last_a;    

	  if (!get_kernel()->has_property(KP_KERNCOMBINATION))
	  {
		  get_kernel()->clear_normal();
		  for(i=0;i<totdoc;i++) {
			  if(a[i] != a_old[i]) {
				  get_kernel()->add_to_normal(regression_fix_index(docs[i]), ((a[i]-a_old[i])*(double)label[i]));
				  a_old[i]=a[i];
			  }
		  }
		  for(i=0;i<totdoc;i++) {
			  if(!shrink_state->active[i]) {
				  lin[i]=shrink_state->last_lin[i]+get_kernel()->compute_optimized(regression_fix_index(docs[i]));
			  }
			  shrink_state->last_lin[i]=lin[i];
		  }
	  }
	  else
		  shrink_state->last_lin[i]=lin[i];
  }
  else if (!use_kernel_cache) 
  {
	  if (!get_kernel()->has_property(KP_KERNCOMBINATION))
		  CIO::message(M_ERROR, "sorry, not implemented") ;
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
		  
		  if ( get_kernel()->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() &&
				  (!get_kernel()->has_property(KP_LINADD) || !get_linadd_enabled()) &&
				  (!((CCombinedKernel*)get_kernel())->get_append_subkernel_weights()) 
			 )
		  {
			  for(ii=0;(i=changed2dnum[ii])>=0;ii++) {

				  CCombinedKernel* k      = (CCombinedKernel*) get_kernel();
				  CKernel* kn = k->get_first_kernel();

				  while (kn)
				  {
					  kn->get_kernel_row(i,inactive2dnum,aicache);

					  for(jj=0;(j=inactive2dnum[jj])>=0;jj++) {
						  kernel_val=aicache[j];
						  lin[j]+=(((a[i]*kernel_val)-(a_old[i]*kernel_val))*(double)label[i]);
					  }

					  kn = k->get_next_kernel(kn);
				  }
			  }
		  }
		  else
		  {
			  for(ii=0;(i=changed2dnum[ii])>=0;ii++) {
				  CKernelMachine::get_kernel()->get_kernel_row(i,inactive2dnum,aicache);
				  for(jj=0;(j=inactive2dnum[jj])>=0;jj++) {
					  kernel_val=aicache[j];
					  lin[j]+=(((a[i]*kernel_val)-(a_old[i]*kernel_val))*(double)label[i]);
				  }
			  }
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

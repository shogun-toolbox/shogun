#include "lib/io.h"
#include "lib/Mathmatics.h"
#include "classifier/svm/SVM_light.h"
#include "classifier/svm/Optimizer.h"
#include "kernel/KernelMachine.h"
#include <assert.h>

CSVMLight::CSVMLight()
{
  model=new MODEL[1];
  learn_parm=new LEARN_PARM[1];
  model->supvec=NULL;
  model->alpha=NULL;
  model->index=NULL;
  set_kernel(NULL);
  primal=NULL;
  dual=NULL;

  //certain setup params
  verbosity=1;
  init_margin=0.15;
  init_iter=500;
  precision_violations=0;
  opt_precision=DEF_PRECISION_LINEAR;
}

CSVMLight::~CSVMLight()
{
  delete[] model->supvec;
  delete[] model->alpha;
  delete[] model->index;
  delete[] model;
  delete[] learn_parm;
  delete[] primal;
  delete[] dual;
}

bool CSVMLight::train()
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
  learn_parm->svm_maxqpsize=50;
  learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize-1;
  learn_parm->maxiter=100000;
  learn_parm->svm_iter_to_shrink=100;
  learn_parm->svm_c=C1;
  learn_parm->eps=-1.0;      /* equivalent regression epsilon for classification */
  learn_parm->transduction_posratio=-1.0;
  learn_parm->svm_costratio=C2/C1;
  learn_parm->svm_costratio_unlab=1.0;
  learn_parm->svm_unlabbound=1E-5;
  learn_parm->epsilon_crit=1E-6; // GU: better decrease it ... ??
  learn_parm->epsilon_a=1E-15;
  learn_parm->compute_loo=0;
  learn_parm->rho=1.0;
  learn_parm->xa_depth=0;
  
  if (!CKernelMachine::get_kernel())
  {
      CIO::message(M_ERROR, "SVM_light can not proceed without kernel!\n");
      return false ;
  }
      
  svm_learn();

  //brain damaged svm light work around
  create_new_model(model->sv_num-1);
  set_bias(-model->b);
  for (INT i=0; i<model->sv_num-1; i++)
  {
	  set_alpha(i, model->alpha[i+1]);
	  set_support_vector(i, model->supvec[i+1]);
  }
  
  return true ;
}

LONG CSVMLight::get_runtime() 
{
  clock_t start;
  start = clock();
  return((LONG)((double)start*100.0/(double)CLOCKS_PER_SEC));
}

void CSVMLight::svm_learn()
{
	LONG *inconsistent, i;
	LONG inconsistentnum;
	LONG misclassified,upsupvecnum;
	double maxdiff, *lin, *c, *a;
	LONG runtime_start,runtime_end;
	LONG iterations;
	LONG trainpos=0, trainneg=0 ;
	INT totdoc=0;
	CLabels* lab=CKernelMachine::get_labels();
	assert(lab!=NULL);
	INT* label=lab->get_int_labels(totdoc);
	assert(label!=NULL);
	LONG* docs=new long[totdoc];

	for (i=0; i<totdoc; i++)
		docs[i]=i;

	double *xi_fullset; /* buffer for storing xi on full sample in loo */
	double *a_fullset;  /* buffer for storing alpha on full sample in loo */
	TIMING timing_profile;
	SHRINK_STATE shrink_state;

	runtime_start=get_runtime();
	timing_profile.time_kernel=0;
	timing_profile.time_opti=0;
	timing_profile.time_shrink=0;
	timing_profile.time_update=0;
	timing_profile.time_model=0;
	timing_profile.time_check=0;
	timing_profile.time_select=0;

	/* make sure -n value is reasonable */
	if((learn_parm->svm_newvarsinqp < 2) 
			|| (learn_parm->svm_newvarsinqp > learn_parm->svm_maxqpsize)) {
		learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;
	}

	init_shrink_state(&shrink_state,totdoc,(LONG)MAXSHRINK);

	inconsistent = new long[totdoc];
	c = new double[totdoc];
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

	delete[] primal;
	delete[] dual;
    primal=new double[learn_parm->svm_maxqpsize*3];
    dual=new double[learn_parm->svm_maxqpsize*2+1];

	for(i=0;i<totdoc;i++) {    /* various inits */
		inconsistent[i]=0;
		c[i]=0;
		a[i]=0;
		lin[i]=0;

		if(label[i] > 0) {
			learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
				fabs((double)label[i]);
			label[i]=1;
			trainpos++;
		}
		else if(label[i] < 0) {
			learn_parm->svm_cost[i]=learn_parm->svm_c*fabs((double)label[i]);
			label[i]=-1;
			trainneg++;
		}
		else {
			learn_parm->svm_cost[i]=0;
		}
	}

  /* compute starting state for initial alpha values */
	CIO::message(M_DEBUG, "alpha:%d num_sv:%d\n", svm_model.alpha, get_num_support_vectors());
  if(svm_model.alpha && get_num_support_vectors()) {
    if(verbosity>=1) {
      printf("Computing starting state..."); fflush(stdout);
    }

	REAL* alpha = new REAL[totdoc];

	for (i=0; i<totdoc; i++)
		alpha[i]=0;

	for (i=0; i<get_num_support_vectors(); i++)
		alpha[get_support_vector(i)]=get_alpha(i);
	
    long* index = new long[totdoc];
    long* index2dnum = new long[totdoc+11];
    REAL* aicache = new REAL[totdoc];
    for(i=0;i<totdoc;i++) {    /* create full index and clip alphas */
      index[i]=1;
      alpha[i]=fabs(alpha[i]);
      if(alpha[i]<0) alpha[i]=0;
      if(alpha[i]>learn_parm->svm_cost[i]) alpha[i]=learn_parm->svm_cost[i];
    }
      for(i=0;i<totdoc;i++)     /* fill kernel cache with unbounded SV */
	if((alpha[i]>0) && (alpha[i]<learn_parm->svm_cost[i]) 
	   && (get_kernel()->kernel_cache_space_available())) 
	  get_kernel()->cache_kernel_row(i);
      for(i=0;i<totdoc;i++)     /* fill rest of kernel cache with bounded SV */
	if((alpha[i]==learn_parm->svm_cost[i]) 
	   && (get_kernel()->kernel_cache_space_available())) 
	  get_kernel()->cache_kernel_row(i);
    (void)compute_index(index,totdoc,index2dnum);
    update_linear_component(docs,label,index2dnum,alpha,a,index2dnum,totdoc,
			    lin,aicache);
    (void)calculate_svm_model(docs,label,lin,alpha,a,c,
			      index2dnum,index2dnum,model);
    for(i=0;i<totdoc;i++) {    /* copy initial alphas */
      a[i]=alpha[i];
    }

    delete[] index;
    delete[] index2dnum;
    delete[] aicache;
    delete[] alpha;

    if(verbosity>=1) {
      printf("done.\n");  fflush(stdout);
    }   
  } 
		CIO::message(M_DEBUG, "%d totdoc %d pos %d neg\n", totdoc, trainpos, trainneg);

	if(verbosity==1) {
		CIO::message(M_MESSAGEONLY, "Optimizing");
	}

	/* train the svm */
  iterations=optimize_to_convergence(docs,label,totdoc,
                     &shrink_state,model,inconsistent,a,lin,
                     c,&timing_profile,
                     &maxdiff,(long)-1,
                     (long)1);


	if(verbosity>=1) {
		if(verbosity==1)
			CIO::message(M_MESSAGEONLY, "done. (%ld iterations)\n",iterations);

		misclassified=0;
		for(i=0;(i<totdoc);i++) { /* get final statistic */
			if((lin[i]-model->b)*(double)label[i] <= 0.0) 
				misclassified++;
		}

		CIO::message(M_INFO, "Optimization finished (%ld misclassified, maxdiff=%.5f).\n",
				misclassified,maxdiff); 

		runtime_end=get_runtime();
		upsupvecnum=0;
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

long CSVMLight::optimize_to_convergence(LONG* docs, INT* label, long int totdoc, 
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
  double eq;
  double *a_old;
  long t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  long transductcycle;
  long transduction;
  double epsilon_crit_org; 
  double bestmaxdiff;
  long   bestmaxdiffiter,terminate;

  double *selcrit;  /* buffer for sorting */        
  REAL *aicache;  /* buffer to keep one row of hessian */
  QP qp;            /* buffer for one quadratic program */

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
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
  terminate=0;
  
  CKernelMachine::get_kernel()->set_time(iteration);  /* for lru cache */
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
  for(;retrain && (!terminate);iteration++) {
  
	  CKernelMachine::get_kernel()->set_time(iteration);  /* for lru cache */

	  CIO::message(M_MESSAGEONLY, ".");

	  if(verbosity>=2) t0=get_runtime();
	  if(verbosity>=3) {
		  CIO::message(M_MESSAGEONLY, "\nSelecting working set... "); 
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
			if(CMath::min(learn_parm->svm_newvarsinqp, learn_parm->svm_maxqpsize-choosenum)>=4) 
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
     
	CKernelMachine::get_kernel()->cache_multiple_kernel_rows(working2dnum, choosenum); 

    if(verbosity>=2) t2=get_runtime();
    if(retrain != 2) {
      optimize_svm(docs,label,inconsistent,0.0,chosen,active2dnum,
		   model,totdoc,working2dnum,choosenum,a,lin,c,
		   aicache,&qp,&epsilon_crit_org);
    }

    if(verbosity>=2) t3=get_runtime();
    update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
			    lin,aicache);

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
      if(verbosity>=1) 
	printf("\nWARNING: Relaxing KT-Conditions due to slow progress! Terminating!\n");
    }

    noshrink=0;
    if((!retrain) && (inactivenum>0) && (!learn_parm->skip_final_opt_check))
	{ 
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
      if((verbosity>=1) || (verbosity>=2)) {
	printf("done.\n");  fflush(stdout);
        printf(" Number of inactive variables = %ld\n",inactivenum);
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
    if(verbosity>=3) {
     CIO::message(M_MESSAGEONLY, "\n");
    }
    
	if (((iteration % 10) == 0) && (!noshrink))
	{
      activenum=shrink_problem(shrink_state,active2dnum,last_suboptimal_at,iteration,totdoc,
			       CMath::max((LONG)(activenum/10),
				    CMath::max((LONG)(totdoc/500),(LONG) 100)),
			       a,inconsistent);
      inactivenum=totdoc-activenum;
      if( (supvecnum>get_kernel()->get_max_elems_cache()) && ((get_kernel()->get_activenum_cache()-activenum)>CMath::max((LONG)(activenum/10),(LONG) 500))) {
	get_kernel()->kernel_cache_shrink(totdoc, CMath::min((LONG) (get_kernel()->get_activenum_cache()-activenum),
				 (LONG) (get_kernel()->get_activenum_cache()-supvecnum)),
			    shrink_state->active); 
      }
    }
  } /* end of loop */

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


void CSVMLight::clear_index(LONG *index)  
              /* initializes and empties index */
{
  index[0]=-1;
} 

void CSVMLight::add_to_index(LONG *index, LONG elem)
     /* initializes and empties index */
{
  register long i;
  for(i=0;index[i] != -1;i++);
  index[i]=elem;
  index[i+1]=-1;
}

LONG CSVMLight::compute_index(LONG *binfeature, LONG range, LONG *index)
     /* create an inverted index of binfeature */
{               
  register long i,ii;

  ii=0;
  for(i=0;i<range;i++) {
    if(binfeature[i]) {
      index[ii]=i;
      ii++;
    }
  }
  for(i=0;i<4;i++) {
    index[ii+i]=-1;
  }
  return(ii);
}


void CSVMLight::optimize_svm(LONG* docs, INT* label,
		  long int *exclude_from_eq_const, double eq_target,
		  long int *chosen, long int *active2dnum, MODEL *model, 
		  long int totdoc, long int *working2dnum, long int varnum, 
		  double *a, double *lin, double *c,
		  REAL *aicache, QP *qp, 
		  double *epsilon_crit_target)
     /* Do optimization on the working set. */
{
    long i;
    double *a_v;

    compute_matrices_for_optimization(docs,label,
				      exclude_from_eq_const,eq_target,chosen,
				      active2dnum,working2dnum,model,a,lin,c,
				      varnum,totdoc,aicache,qp);

    if(verbosity>=3) {
     CIO::message(M_DEBUG, "Running optimizer...");
    }
    /* call the qp-subsolver */
    a_v=optimize_qp(qp,epsilon_crit_target,
		    learn_parm->svm_maxqpsize,
		    &(model->b),  				/* in case the optimizer gives us */
            learn_parm->svm_maxqpsize); /* the threshold for free. otherwise */
                                   		/* b is calculated in calculate_model. */
    if(verbosity>=3) {         
     CIO::message(M_DEBUG, "done\n");
    }

    for(i=0;i<varnum;i++) {
      a[working2dnum[i]]=a_v[i];
    }
}

void CSVMLight::compute_matrices_for_optimization(LONG* docs, INT* label, 
          long *exclude_from_eq_const, double eq_target,
	  long int *chosen, long int *active2dnum, 
          long int *key, MODEL *model, double *a, double *lin, double *c, 
	  long int varnum, long int totdoc,
          REAL *aicache, QP *qp)
{
  register long ki,kj,i,j;
  register double kernel_temp;

  qp->opt_n=varnum;
  qp->opt_ce0[0]=-eq_target; /* compute the constant for equality constraint */
  for(j=1;j<model->sv_num;j++) { /* start at 1 */
    if((!chosen[model->supvec[j]])
       && (!exclude_from_eq_const[(model->supvec[j])])) {
      qp->opt_ce0[0]+=model->alpha[j];
    }
  } 
  if(learn_parm->biased_hyperplane) 
    qp->opt_m=1;
  else 
    qp->opt_m=0;  /* eq-constraint will be ignored */

  /* init linear part of objective function */
  for(i=0;i<varnum;i++) {
    qp->opt_g0[i]=lin[key[i]];
  }

  for(i=0;i<varnum;i++) {
    ki=key[i];

    /* Compute the matrix for equality constraints */
    qp->opt_ce[i]=label[ki];
    qp->opt_low[i]=0;
    qp->opt_up[i]=learn_parm->svm_cost[ki];

    kernel_temp=(double)CKernelMachine::get_kernel()->kernel(docs[ki], docs[ki]); 
    /* compute linear part of objective function */
    qp->opt_g0[i]-=(kernel_temp*a[ki]*(double)label[ki]); 
    /* compute quadratic part of objective function */
    qp->opt_g[varnum*i+i]=kernel_temp;
    for(j=i+1;j<varnum;j++) {
      kj=key[j];
      kernel_temp=(double)CKernelMachine::get_kernel()->kernel(docs[ki], docs[kj]);
      /* compute linear part of objective function */
      qp->opt_g0[i]-=(kernel_temp*a[kj]*(double)label[kj]);
      qp->opt_g0[j]-=(kernel_temp*a[ki]*(double)label[ki]); 
      /* compute quadratic part of objective function */
      qp->opt_g[varnum*i+j]=(double)label[ki]*(double)label[kj]*kernel_temp;
      qp->opt_g[varnum*j+i]=(double)label[ki]*(double)label[kj]*kernel_temp;
    }

    if(verbosity>=3) {
      if(i % 20 == 0) {
	CIO::message(M_DEBUG, "%ld..",i);
      }
    }
  }

  for(i=0;i<varnum;i++) {
    /* assure starting at feasible point */
    qp->opt_xinit[i]=a[key[i]];
    /* set linear part of objective function */
    qp->opt_g0[i]=(learn_parm->eps-(double)label[key[i]]*c[key[i]])+qp->opt_g0[i]*(double)label[key[i]];    
  }

  if(verbosity>=3) {
    CIO::message(M_DEBUG, "done\n");
  }
}

long CSVMLight::calculate_svm_model(LONG* docs, INT *label,
			 double *lin, double *a, double *a_old, double *c, 
			 long int *working2dnum, long int *active2dnum, MODEL *model)
     /* Compute decision function based on current values */
     /* of alpha. */
{
  long i,ii,pos,b_calculated=0,first_low,first_high;
  double ex_c,b_temp,b_low,b_high;

  if(verbosity>=3) {
   CIO::message(M_DEBUG, "Calculating model...");
  }

  if(!learn_parm->biased_hyperplane) {
    model->b=0;
    b_calculated=1;
  }

  for(ii=0;(i=working2dnum[ii])>=0;ii++) {
    if((a_old[i]>0) && (a[i]==0)) { /* remove from model */
      pos=model->index[i]; 
      model->index[i]=-1;
      (model->sv_num)--;
      model->supvec[pos]=model->supvec[model->sv_num];
      model->alpha[pos]=model->alpha[model->sv_num];
      model->index[model->supvec[pos]]=pos;
    }
    else if((a_old[i]==0) && (a[i]>0)) { /* add to model */
      model->supvec[model->sv_num]=docs[i];
      model->alpha[model->sv_num]=a[i]*(double)label[i];
      model->index[i]=model->sv_num;
      (model->sv_num)++;
    }
    else if(a_old[i]==a[i]) { /* nothing to do */
    }
    else {  /* just update alpha */
      model->alpha[model->index[i]]=a[i]*(double)label[i];
    }
      
    ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
    if((a_old[i]>=ex_c) && (a[i]<ex_c)) { 
      (model->at_upper_bound)--;
    }
    else if((a_old[i]<ex_c) && (a[i]>=ex_c)) { 
      (model->at_upper_bound)++;
    }

    if((!b_calculated) 
       && (a[i]>learn_parm->epsilon_a) && (a[i]<ex_c)) {   /* calculate b */
     	model->b=((double)label[i]*learn_parm->eps-c[i]+lin[i]); 
	b_calculated=1;
    }
  }      

  /* No alpha in the working set not at bounds, so b was not
     calculated in the usual way. The following handles this special
     case. */
  if(learn_parm->biased_hyperplane 
     && (!b_calculated)
     && (model->sv_num-1 == model->at_upper_bound)) { 
    first_low=1;
    first_high=1;
    b_low=0;
    b_high=0;
    for(ii=0;(i=active2dnum[ii])>=0;ii++) {
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if(a[i]<ex_c) { 
	if(label[i]>0)  {
	  b_temp=-(learn_parm->eps-c[i]+lin[i]);
	  if((b_temp>b_low) || (first_low)) {
	    b_low=b_temp;
	    first_low=0;
	  }
	}
	else {
	  b_temp=-(-learn_parm->eps-c[i]+lin[i]);
	  if((b_temp<b_high) || (first_high)) {
	    b_high=b_temp;
	    first_high=0;
	  }
	}
      }
      else {
	if(label[i]<0)  {
	  b_temp=-(-learn_parm->eps-c[i]+lin[i]);
	  if((b_temp>b_low) || (first_low)) {
	    b_low=b_temp;
	    first_low=0;
	  }
	}
	else {
	  b_temp=-(learn_parm->eps-c[i]+lin[i]);
	  if((b_temp<b_high) || (first_high)) {
	    b_high=b_temp;
	    first_high=0;
	  }
	}
      }
    }
    if(first_high) {
      model->b=-b_low;
    }
    else if(first_low) {
      model->b=-b_high;
    }
    else {
      model->b=-(b_high+b_low)/2.0;  /* select b as the middle of range */
      /* printf("\nb_low=%f, b_high=%f,b=%f\n",b_low,b_high,model->b); */
    }
  }

  if(verbosity>=3) {
   CIO::message(M_DEBUG, "done\n");
  }

  return(model->sv_num-1); /* have to substract one, since element 0 is empty*/
}

long CSVMLight::check_optimality(MODEL *model, INT* label,
		      double *a, double *lin, double *c, long int totdoc, 
		      double *maxdiff, double epsilon_crit_org, long int *misclassified, 
		      long int *inconsistent, long int *active2dnum,
		      long int *last_suboptimal_at, 
		      long int iteration)
     /* Check KT-conditions */
{
  long i,ii,retrain;
  double dist,ex_c,target;

  learn_parm->epsilon_shrink=learn_parm->epsilon_shrink*0.7+(*maxdiff)*0.3; 
  retrain=0;
  (*maxdiff)=0;
  (*misclassified)=0;
  for(ii=0;(i=active2dnum[ii])>=0;ii++) {
    if((!inconsistent[i]) && label[i]) {
      dist=(lin[i]-model->b)*(double)label[i];/* 'distance' from
						 hyperplane*/
      target=-(learn_parm->eps-(double)label[i]*c[i]);
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if(dist <= 0) {       
	(*misclassified)++;  /* does not work due to deactivation of var */
      }
      if((a[i]>learn_parm->epsilon_a) && (dist > target)) {
	if((dist-target)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-target;
      }
      else if((a[i]<ex_c) && (dist < target)) {
	if((target-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=target-dist;
      }
      /* Count how long a variable was at lower/upper bound (and optimal).*/
      /* Variables, which were at the bound and optimal for a long */
      /* time are unlikely to become support vectors. In case our */
      /* cache is filled up, those variables are excluded to save */
      /* kernel evaluations. (See chapter 'Shrinking').*/ 
      if((a[i]>(learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	last_suboptimal_at[i]=iteration;         /* not at bound */
      }
      else if((a[i]<=(learn_parm->epsilon_a)) 
	      && (dist < (target+learn_parm->epsilon_shrink))) {
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
      else if((a[i]>=ex_c)
	      && (dist > (target-learn_parm->epsilon_shrink)))  { 
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
    }   
  }
  /* termination criterion */
  if((!retrain) && ((*maxdiff) > learn_parm->epsilon_crit)) {  
    retrain=1;
  }
  return(retrain);
}

void CSVMLight::update_linear_component(LONG* docs, INT* label, 
			     long int *active2dnum, double *a, 
			     double *a_old, long int *working2dnum, 
			     long int totdoc,
			     double *lin, REAL *aicache)
     /* keep track of the linear component */
     /* lin of the gradient etc. by updating */
     /* based on the change of the variables */
     /* in the current working set */
{
  register long i,ii,j,jj;
  register double tec;

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

/*************************** Working set selection ***************************/

long CSVMLight::select_next_qp_subproblem_grad(INT* label, 
				    double *a, double *lin, 
				    double *c, long int totdoc, 
				    long int qp_size, 
				    long int *inconsistent, 
				    long int *active2dnum, 
				    long int *working2dnum, 
				    double *selcrit, 
				    long int *select, 
				    long int cache_only,
				    long int *key, long int *chosen)
     /* Use the feasible direction approach to select the next
      qp-subproblem (see chapter 'Selecting a good working set'). If
      'cache_only' is true, then the variables are selected only among
      those for which the kernel evaluations are cached. */
{
  long choosenum,i,j,k,activedoc,inum,valid;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if(cache_only) 
      valid=(get_kernel()->kernel_cache_check(j));
    else
      valid=1;
    if(valid
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j]))
      {
      selcrit[activedoc]=(double)label[j]*(learn_parm->eps-(double)label[j]*c[j]+(double)label[j]*lin[j]);
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
      i=key[select[k]];
      chosen[i]=1;
      working2dnum[inum+choosenum]=i;
      choosenum+=1;
	CKernelMachine::get_kernel()->kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if(cache_only) 
      valid=(get_kernel()->kernel_cache_check(j));
    else
      valid=1;
    if(valid
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j])) 
      {
      selcrit[activedoc]=-(double)label[j]*(learn_parm->eps-(double)label[j]*c[j]+(double)label[j]*lin[j]);
      /*  selcrit[activedoc]=-(double)(label[j]*(-1.0+(double)label[j]*lin[j])); */
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
	CKernelMachine::get_kernel()->kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

long CSVMLight::select_next_qp_subproblem_rand(INT* label, 
				    double *a, double *lin, 
				    double *c, long int totdoc, 
				    long int qp_size, 
				    long int *inconsistent, 
				    long int *active2dnum, 
				    long int *working2dnum, 
				    double *selcrit, 
				    long int *select, 
				    long int *key, 
				    long int *chosen, 
				    long int iteration)
/* Use the feasible direction approach to select the next
   qp-subproblem (see section 'Selecting a good working set'). Chooses
   a feasible direction at (pseudo) random to help jump over numerical
   problem. */
{
  long choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if((!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=(j+iteration) % totdoc;
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
	CKernelMachine::get_kernel()->kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if((!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=(j+iteration) % totdoc;
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
	CKernelMachine::get_kernel()->kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}



void CSVMLight::select_top_n(double *selcrit, long range, long* select,
		  long int n)
{
  register long i,j;

  for(i=0;(i<n) && (i<range);i++) { /* Initialize with the first n elements */
    for(j=i;j>=0;j--) {
      if((j>0) && (selcrit[select[j-1]]<selcrit[i])){
	select[j]=select[j-1];
      }
      else {
	select[j]=i;
	j=-1;
      }
    }
  }
  if(n>0) {
    for(i=n;i<range;i++) {  
      if(selcrit[i]>selcrit[select[n-1]]) {
	for(j=n-1;j>=0;j--) {
	  if((j>0) && (selcrit[select[j-1]]<selcrit[i])) {
	    select[j]=select[j-1];
	  }
	  else {
	    select[j]=i;
	    j=-1;
	  }
	}
      }
    }
  }
}      
      

/******************************** Shrinking  *********************************/

void CSVMLight::init_shrink_state(SHRINK_STATE *shrink_state, long int totdoc,
		       long int maxhistory)
{
  long i;

  shrink_state->deactnum=0;
  shrink_state->active = new long[totdoc];
  shrink_state->inactive_since = new long[totdoc];
  shrink_state->a_history = new double*[maxhistory];
  shrink_state->maxhistory=maxhistory;
  shrink_state->last_lin = new double[totdoc];
  shrink_state->last_a = new double[totdoc];

  for(i=0;i<totdoc;i++) { 
    shrink_state->active[i]=1;
    shrink_state->inactive_since[i]=0;
    shrink_state->last_a[i]=0;
    shrink_state->last_lin[i]=0;
  }
}

void CSVMLight::shrink_state_cleanup(SHRINK_STATE *shrink_state)
{
  delete[] shrink_state->active;
  delete[] shrink_state->inactive_since;
  if(shrink_state->deactnum > 0) 
    delete[] (shrink_state->a_history[shrink_state->deactnum-1]);
  delete[] (shrink_state->a_history);
  delete[] (shrink_state->last_a);
  delete[] (shrink_state->last_lin);
}

long CSVMLight::shrink_problem(SHRINK_STATE *shrink_state, 
		    long int *active2dnum, 
		    long int *last_suboptimal_at, 
		    long int iteration, 
		    long int totdoc, 
		    long int minshrink, 
		    double *a, 
		    long int *inconsistent)
     /* Shrink some variables away.  Do the shrinking only if at least
        minshrink variables can be removed. */
{
  long i,ii,change,activenum,lastiter;
  double *a_old=NULL;
  
  activenum=0;
  change=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    activenum++;
      lastiter=last_suboptimal_at[i];
    if(((iteration-lastiter) > learn_parm->svm_iter_to_shrink) 
       || (inconsistent[i])) {
      change++;
    }
  }
  if((change>=minshrink) /* shrink only if sufficiently many candidates */
     && (shrink_state->deactnum<shrink_state->maxhistory)) { /* and enough memory */
    /* Shrink problem by removing those variables which are */
    /* optimal at a bound for a minimum number of iterations */
    if(verbosity>=2) {
     CIO::message(M_INFO, " Shrinking...");
    }
    a_old=new double[totdoc];
    shrink_state->a_history[shrink_state->deactnum]=a_old;
    for(i=0;i<totdoc;i++) {
      a_old[i]=a[i];
    }
    for(ii=0;active2dnum[ii]>=0;ii++) {
      i=active2dnum[ii];
	lastiter=last_suboptimal_at[i];
      if(((iteration-lastiter) > learn_parm->svm_iter_to_shrink) 
	 || (inconsistent[i])) {
	shrink_state->active[i]=0;
	shrink_state->inactive_since[i]=shrink_state->deactnum;
      }
    }
    activenum=compute_index(shrink_state->active,totdoc,active2dnum);
    shrink_state->deactnum++;
    if(verbosity>=2) {
     CIO::message(M_INFO, "done.\n");
     CIO::message(M_INFO, " Number of inactive variables = %ld\n",totdoc-activenum);
    }
  }
  return(activenum);
} 

void CSVMLight::reactivate_inactive_examples(INT* label, 
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
  register long i,j,ii,jj,t,*changed2dnum,*inactive2dnum;
  long *changed,*inactive;
  register double kernel_val,*a_old,dist;
  double ex_c,target;

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
	for(jj=0;(j=inactive2dnum[jj])>=0;jj++) {
	  kernel_val=aicache[j];
	  lin[j]+=(((a[i]*kernel_val)-(a_old[i]*kernel_val))*(double)label[i]);
	}
      }
  }
  delete[] changed;
  delete[] changed2dnum;
  delete[] inactive;
  delete[] inactive2dnum;

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
  for(i=0;i<totdoc;i++) {
    (shrink_state->a_history[shrink_state->deactnum-1])[i]=a[i];
  }
  for(t=shrink_state->deactnum-2;(t>=0) && shrink_state->a_history[t];t--) {
      delete[] shrink_state->a_history[t];
      shrink_state->a_history[t]=0;
  }
}

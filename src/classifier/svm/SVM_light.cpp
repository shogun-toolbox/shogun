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
  //CKernelMachine::set_kernel(NULL);
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
  strcpy (learn_parm->alphafile, "");
  learn_parm->biased_hyperplane=1;
  learn_parm->remove_inconsistent=0;
  learn_parm->skip_final_opt_check=1;
  learn_parm->svm_maxqpsize=50;
  learn_parm->svm_newvarsinqp=0;
  learn_parm->svm_iter_to_shrink=100;
  learn_parm->svm_c=C1;
  learn_parm->transduction_posratio=0.5;
  learn_parm->svm_costratio=1.0;
  learn_parm->svm_costratio_unlab=1.0;
  learn_parm->svm_unlabbound=1E-5;
  learn_parm->epsilon_crit=1E-6; // GU: better decrease it ... ??
  learn_parm->epsilon_a=1E-15;
  learn_parm->compute_loo=0;
  learn_parm->rho=1.0;
  learn_parm->xa_depth=0;
  
  if (!CKernelMachine::get_kernel())
  {
      CIO::message("SVM_light can not proceed without kernel!\n");
      return false ;
  }
      
  svm_learn();

  //brain damaged svm light work around
  create_new_model(model->sv_num-1);
  set_bias(model->b);
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
	LONG *inconsistent, i; //, *unlabeled
	LONG inconsistentnum;
	LONG misclassified,upsupvecnum;
	//  double loss,model_length,example_length;
	double maxdiff,*lin,*a;
	LONG runtime_start,runtime_end;
	LONG iterations;
	//  LONG heldout;
	//  LONG loo_count=0,loo_count_pos=0,loo_count_neg=0;
	LONG trainpos=0, trainneg=0 ;
	//  LONG loocomputed=0,runtime_start_loo=0,runtime_start_xa=0;
	//  double heldout_c=0;
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

	init_shrink_state(&shrink_state,totdoc,(LONG)10000);

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

	delete[] primal;
	delete[] dual;
    primal=new double[learn_parm->svm_maxqpsize*3];
    dual=new double[learn_parm->svm_maxqpsize*2+1];

	for(i=0;i<totdoc;i++) {    /* various inits */
		inconsistent[i]=0;
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

	if(learn_parm->remove_inconsistent && learn_parm->compute_loo) {
		learn_parm->compute_loo=0;
		CIO::message("\nCannot compute leave-one-out estimates when removing inconsistent examples.\n\n");
	}    

	if((trainpos == 1) || (trainneg == 1)) {
		learn_parm->compute_loo=0;
		CIO::message("\nCannot compute leave-one-out with only one example in one class.\n\n");
	}    


	if(verbosity==1) {
		CIO::message("Optimizing");
	}

	/* train the svm */
	iterations=optimize_to_convergence(docs,label,totdoc,
			&shrink_state,model,inconsistent,a,lin,&timing_profile,
			&maxdiff,(LONG)-1,(LONG)1);

	if(verbosity>=1) {
		if(verbosity==1)
			CIO::message("done. (%ld iterations)\n",iterations);

		misclassified=0;
		for(i=0;(i<totdoc);i++) { /* get final statistic */
			if((lin[i]+model->b)*(double)label[i] <= 0.0) 
				misclassified++;
		}

		CIO::message("Optimization finished (%ld misclassified, maxdiff=%.5f).\n",
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
		CIO::message("Number of SV: %ld (including %ld at upper bound)\n",
				model->sv_num-1,upsupvecnum);
	}

	/*  if(learn_parm->alphafile[0])
		write_alphas(learn_parm->alphafile,a,label,totdoc);*/

	shrink_state_cleanup(&shrink_state);
	delete[] label;
	delete[] inconsistent;
	delete[] a;
	delete[] a_fullset;
	delete[] xi_fullset;
	delete[] lin;
	delete[] learn_parm->svm_cost;
	delete[] docs;
}

LONG CSVMLight::optimize_to_convergence(
LONG* docs,                 /* Training vectors (x-part) */
INT* label,               /* Training labels (y-part) */
LONG totdoc,               /* Number of examples in docs/label */
SHRINK_STATE *shrink_state,/* State of active variables */
MODEL *model,              /* Returns learning result */
LONG *inconsistent,
double *a,
double *lin,
TIMING *timing_profile,
double *maxdiff,
LONG heldout,
LONG retrain)
{
  LONG *chosen,*key,i,j,jj,*last_suboptimal_at,noshrink;
  LONG inconsistentnum,choosenum,already_chosen=0,iteration;
  LONG misclassified,supvecnum=0,*active2dnum,inactivenum;
  LONG *working2dnum,*selexam;
  LONG activenum;
  double criterion,eq;
  double *a_old=NULL;
  LONG t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  LONG transductcycle;
  double epsilon_crit_org; 

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
  if(!retrain) retrain=1;
  iteration=1;
  
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
  for(;retrain && iteration < 1000000 ;iteration++) 
  {
  
	  CKernelMachine::get_kernel()->set_time(iteration);  /* for lru cache */

	  CIO::message(".");

	  if(verbosity>=2) t0=get_runtime();
	  if(verbosity>=3) {
		  CIO::message("\nSelecting working set... "); 
	  }

    i=0;
    for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear working set */
      if((chosen[j]>=(learn_parm->svm_maxqpsize/
		      math.min(learn_parm->svm_maxqpsize,
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
	for(i=0;i<totdoc;i++) { /* make sure we fulfill equality constraINT */
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
	{      /* select working set according to steepest gradient */ 
		if (math.min(learn_parm->svm_newvarsinqp,learn_parm->svm_maxqpsize)>=4)
		{
			/* select part of the working set from cache */
			already_chosen=select_next_qp_subproblem_grad_cache(
					label,a,lin,totdoc,
					math.min((LONG)(learn_parm->svm_maxqpsize-choosenum),
						(LONG)(learn_parm->svm_newvarsinqp/2)),
					inconsistent,active2dnum,
					working2dnum,selcrit,selexam,
					key,chosen);
			choosenum+=already_chosen;
		}
		choosenum+=select_next_qp_subproblem_grad(label,a,lin,totdoc,
				math.min((LONG)(learn_parm->svm_maxqpsize-choosenum),
					(LONG)(learn_parm->svm_newvarsinqp-already_chosen)),
				inconsistent,active2dnum,
				working2dnum,selcrit,selexam,key,
				chosen);
	}

    if(verbosity>=2) {
     CIO::message(" %ld vectors chosen\n",choosenum); 
    }

    if(verbosity>=2) t1=get_runtime();
     
	CKernelMachine::get_kernel()->cache_multiple_kernel_rows(working2dnum, choosenum); 

    if(verbosity>=2) t2=get_runtime();
    if(retrain != 2) {
      optimize_svm(docs,label,chosen,active2dnum,model,totdoc,
		   working2dnum,choosenum,a,lin,aicache,
		   &qp,&epsilon_crit_org);
    }

    if(verbosity>=2) t3=get_runtime();
    update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
			    lin,aicache,
				NULL);

    if(verbosity>=2) t4=get_runtime();
    supvecnum=calculate_svm_model(docs,label,lin,a,a_old, working2dnum,model);

    if(verbosity>=2) t5=get_runtime();

    /* The following computation of the objective function works only */
    /* relative to the active variables */
    if(verbosity>=3) {
      criterion=compute_objective_function(a,lin,label,active2dnum);
     CIO::message("Objective function (over active variables): %.16f\n",criterion);
      
    }

    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      a_old[i]=a[i];
    }

    retrain=check_optimality(model,label,a,lin,totdoc,
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

    noshrink=0;

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
     CIO::message(" => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
	     supvecnum,model->at_upper_bound,(*maxdiff)); 
     
    }
    if(verbosity>=3) {
     CIO::message("\n");
    }

    if(((iteration % 10) == 0) && (!noshrink)) {
      activenum=shrink_problem(learn_parm,shrink_state,active2dnum,last_suboptimal_at,
			       iteration,totdoc,math.max((LONG)(activenum/10),(LONG) 100),
			       a,inconsistent);
      inactivenum=totdoc-activenum;
    }
  } /* end of loop */

  if(verbosity>=1) {
	  criterion=compute_objective_function(a,lin,label,active2dnum);
	  CIO::message("\nobj = %.16f\n",criterion);
  }

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

/* Return value of objective function. */
/* Works only relative to the active variables! */
double CSVMLight::compute_objective_function(double *a, double* lin, INT *label, LONG* active2dnum)
{
  LONG i,ii;
  double criterion;
  /* calculate value of objective function */
  criterion=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    criterion=criterion-a[i]+0.5*a[i]*label[i]*lin[i];
  } 
  return(criterion);
}

/* initializes and empties index */
void CSVMLight::clear_index(LONG *index)  
{
  index[0]=-1;
} 

/* initializes and empties index */
void CSVMLight::add_to_index(LONG *index, LONG elem)
{
  register LONG i;
  for(i=0;index[i] != -1;i++);
  index[i]=elem;
  index[i+1]=-1;
}

/* create an inverted index of binfeature */
LONG CSVMLight::compute_index(LONG *binfeature, LONG range, LONG *index)
{               
  register LONG i,ii;

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

void CSVMLight::optimize_svm(
LONG* docs,               /* Do optimization on the working set. */
INT* label, LONG *chosen, LONG *active2dnum,
MODEL *model,
LONG totdoc, LONG* working2dnum, LONG varnum,
double *a, double* lin,
REAL *aicache,
QP *qp,
double *epsilon_crit_target)
{
    LONG i;
    double *a_v;

    compute_matrices_for_optimization(docs,label,chosen,active2dnum,
				      working2dnum,model,a,lin,varnum,
				      totdoc,aicache,
				      qp);

    if(verbosity>=3) {
     CIO::message("Running optimizer...");
    }

    /* call the qp-subsolver */
	/* in case the optimizer gives us */
	/* the threshold for free. otherwise */
	/* b is calculated in calculate_model. */
    a_v=optimize_qp(qp,epsilon_crit_target,
		    learn_parm->svm_maxqpsize,
		    &(model->b), primal, dual,
			init_margin, init_iter, precision_violations,
			model_b, opt_precision);

    if(verbosity>=3) {         
     CIO::message("done\n");
    }

    for(i=0;i<varnum;i++) {
      a[working2dnum[i]]=a_v[i];
      /*
      if(a_v[i]<=(0+learn_parm->epsilon_a)) {
	a[working2dnum[i]]=0;
      }
      else if(a_v[i]>=(learn_parm->svm_cost[working2dnum[i]]-learn_parm->epsilon_a)) {
	a[working2dnum[i]]=learn_parm->svm_cost[working2dnum[i]];
      }
      */
    }
}

void CSVMLight::compute_matrices_for_optimization(
LONG *docs,
INT *label,LONG* chosen,LONG* active2dnum, LONG* key,
MODEL *model,
double *a, double* lin,
LONG varnum, LONG totdoc,
REAL *aicache,
QP *qp)
{
  register LONG ki,kj,i,j;
  register double kernel_temp;

  qp->opt_n=varnum;
  qp->opt_ce0[0]=0; /* compute the constant for equality constraINT */
  for(j=1;j<model->sv_num;j++) { /* start at 1 */
    if(!chosen[model->supvec[j]]) {
      qp->opt_ce0[0]+=model->alpha[j];
    }
  } 
  if(learn_parm->biased_hyperplane) 
    qp->opt_m=1;
  else 
    qp->opt_m=0;  /* eq-constraINT will be ignored */

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
	CIO::message("%ld..",i);
      }
    }
  }

  for(i=0;i<varnum;i++) {
    /* assure starting at feasible poINT */
    qp->opt_xinit[i]=a[key[i]];
    /* set linear part of objective function */
    qp->opt_g0[i]=-1.0+qp->opt_g0[i]*(double)label[key[i]];    
  }

  if(verbosity>=3) {
    CIO::message("done\n");
  }
}

LONG CSVMLight::calculate_svm_model(
LONG *docs,              /* Compute decision function based on current values */
INT *label, /* of alpha. */
double *lin, double *a, double* a_old,
LONG *working2dnum,
MODEL *model)
{
  LONG i,ii,pos,b_calculated=0;
  double ex_c;

  if(verbosity>=3) {
   CIO::message("Calculating model...");
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
	model->b=((double)label[i]-lin[i]);
	b_calculated=1;
    }
  }      

  if(verbosity>=3) {
   CIO::message("done\n");
  }

  /* If there is no alpha in the working set not at bounds, then
     just use the model->b from the last iteration or the one provided 
     by the core optimizer */

  return(model->sv_num-1); /* have to substract one, since element 0 is empty*/
}

/* Check KT-conditions */
LONG CSVMLight::check_optimality(
MODEL *model,            
INT *label,
double *a, double* lin,
LONG totdoc,
double *maxdiff, double epsilon_crit_org,
LONG *misclassified,
LONG *inconsistent,LONG* active2dnum, LONG *last_suboptimal_at, LONG iteration)
{
  LONG i,ii,retrain;
  double dist,ex_c;

  learn_parm->epsilon_shrink=learn_parm->epsilon_shrink*0.7+(*maxdiff)*0.3; 
  retrain=0;
  (*maxdiff)=0;
  (*misclassified)=0;
  for(ii=0;(i=active2dnum[ii])>=0;ii++) {
    if((!inconsistent[i]) && label[i]) {
      dist=(lin[i]+model->b)*(double)label[i];/* 'distance' from hyperplane*/
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if(dist <= 0) {       
	(*misclassified)++;  /* does not work due to deactivation of var */
      }
      if((a[i]>learn_parm->epsilon_a) && (dist > 1)) {
	if((dist-1.0)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-1.0;
      }
      else if((a[i]<ex_c) && (dist < 1)) {
	if((1.0-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=1.0-dist;
      }
      /* Count how LONG a variable was at lower/upper bound (and optimal).*/
      /* Variables, which were at the bound and optimal for a LONG */
      /* time are unlikely to become support vectors. In case our */
      /* cache is filled up, those variables are excluded to save */
      /* kernel evaluations. (See chapter 'Shrinking').*/ 
      if((a[i]>(learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	last_suboptimal_at[i]=iteration;         /* not at bound */
      }
      else if((a[i]<=(learn_parm->epsilon_a)) 
	      && (dist < (1.0+learn_parm->epsilon_shrink))) {
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
      else if((a[i]>=ex_c)
	      && (dist > (1.0-learn_parm->epsilon_shrink)))  { 
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

LONG CSVMLight::identify_misclassified(
double *lin,
INT *label, LONG totdoc,
MODEL *model,
LONG *inconsistentnum, LONG* inconsistent)
{
  LONG i,retrain;
  double dist;

  /* Throw out misclassified examples. This */
  /* corresponds to the -i 2 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(i=0;i<totdoc;i++) {
    dist=(lin[i]+model->b)*(double)label[i]; /* 'distance' from hyperplane*/  
  }
  return(retrain);
}

LONG CSVMLight::identify_one_misclassified(
double *lin,
INT *label, LONG totdoc,
MODEL *model,
LONG *inconsistentnum, LONG *inconsistent)
{
  LONG i=-1,retrain,maxex=-1;
  //  double dist,maxdist=0;

  /* Throw out the 'most misclassified' example. This */
  /* corresponds to the -i 3 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  
  if(maxex>=0) {
    (*inconsistentnum)++;
    inconsistent[maxex]=1;  /* never choose again */
    retrain=2;          /* start over */
    if(verbosity>=3) {
     CIO::message("inconsistent(%ld)..",i);
    }
  }
  return(retrain);
}

void CSVMLight::update_linear_component(
LONG* docs,
INT *label, 
LONG *active2dnum,                  /* keep track of the linear component */
double *a, double* a_old,          /* lin of the gradient etc. by updating */
LONG *working2dnum, LONG totdoc, /* based on the change of the variables */
double *lin,
REAL *aicache,
double *weights)
{
	register LONG i,ii,j,jj;
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

LONG CSVMLight::select_next_qp_subproblem_grad(
INT *label,
double *a,double* lin,      /* Use the feasible direction approach to select the */
LONG totdoc, LONG qp_size, /* next qp-subproblem  (see section 'Selecting a good */
LONG *inconsistent, LONG* active2dnum, LONG* working2dnum,
double *selcrit,
LONG *select,
LONG *key, LONG *chosen)
{
  LONG choosenum,i,j,k,activedoc,inum;
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
      selcrit[activedoc]=lin[j]-(double)label[j];
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(LONG)(qp_size/2));
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
      selcrit[activedoc]=(double)(label[j])-lin[j];
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(LONG)(qp_size/2));
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


LONG CSVMLight::select_next_qp_subproblem_grad_cache(
INT *label,
double *a, double *lin,         // Use the feasible direction approach to select the 
LONG totdoc, LONG qp_size,    // next qp-subproblem  (see chapter 'Selecting a 
LONG *inconsistent, LONG* active2dnum, LONG* working2dnum, // cached kernel 
double *selcrit,
LONG *select,
LONG *key, LONG* chosen)
{
return CSVMLight::select_next_qp_subproblem_grad(
label,a,lin,totdoc,qp_size,inconsistent,active2dnum,working2dnum,selcrit,select,
key,chosen);
}
/*
  LONG choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); // find end of index
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if((kernel_cache->index[j]>=0)
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j]))
      {
      selcrit[activedoc]=(double)label[j]*(-1.0+(double)label[j]*lin[j]);
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(LONG)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if((kernel_cache->index[j]>=0)
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j])) 
      {
      selcrit[activedoc]=-(double)(label[j]*(-1.0+(double)label[j]*lin[j]));
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(LONG)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
  } 
  working2dnum[inum+choosenum]=-1; // complete index
  return(choosenum);
}
*/

void CSVMLight::select_top_n(double *selcrit, LONG range,LONG* select, LONG n)
{
  register LONG i,j;

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
      

/******************************** Shrinking  *********************************/

void CSVMLight::init_shrink_state(SHRINK_STATE *shrink_state, LONG totdoc, LONG maxhistory)
{
  LONG i;

  shrink_state->deactnum=0;
  shrink_state->active = new long[totdoc];
  shrink_state->inactive_since = new long[totdoc];
  shrink_state->a_history = new double*[10000];

  for(i=0;i<totdoc;i++) { 
    shrink_state->active[i]=1;
    shrink_state->inactive_since[i]=0;
  }
}

void CSVMLight::shrink_state_cleanup(SHRINK_STATE *shrink_state)
{
  delete[] shrink_state->active;
  delete[] shrink_state->inactive_since;
  for (int i=0; i<shrink_state->deactnum; i++)
    delete[] shrink_state->a_history[i];
  delete[] shrink_state->a_history;
}

/* shrink some variables away */
/* do the shrinking only if at least minshrink variables can be removed */
LONG CSVMLight::shrink_problem(LEARN_PARM *learn_parm, SHRINK_STATE *shrink_state, LONG *active2dnum, 
					LONG *last_suboptimal_at, LONG iteration, LONG totdoc, LONG minshrink, 
					double *a, LONG *inconsistent)
{
  LONG i,ii,change,activenum;
  double *a_old=NULL;
  
  activenum=0;
  change=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    activenum++;
    if(((iteration-last_suboptimal_at[i])
	> learn_parm->svm_iter_to_shrink) 
       || (inconsistent[i])) {
      change++;
    }
  }
  if(change>=minshrink) { /* shrink only if sufficiently many candidates */
    /* Shrink problem by removing those variables which are */
    /* optimal at a bound for a minimum number of iterations */
    if(verbosity>=2) {
     CIO::message(" Shrinking...");
    }
    a_old=new double[totdoc];
    shrink_state->a_history[shrink_state->deactnum]=a_old;
    for(i=0;i<totdoc;i++) {
      a_old[i]=a[i];
    }
    change=0;
    for(ii=0;active2dnum[ii]>=0;ii++) {
      i=active2dnum[ii];
      if((((iteration-last_suboptimal_at[i])
	   >learn_parm->svm_iter_to_shrink) 
	  || (inconsistent[i]))) {
	shrink_state->active[i]=0;
	shrink_state->inactive_since[i]=shrink_state->deactnum;
	change++;
      }
    }
    activenum=compute_index(shrink_state->active,totdoc,active2dnum);
    shrink_state->deactnum++;
    if(verbosity>=2) {
     CIO::message("done.\n");
     CIO::message(" Number of inactive variables = %ld\n",totdoc-activenum);
    }
  }
  //  delete[] a_old ;
  //  a_old=NULL ;
  return(activenum);
} 


void CSVMLight::reactivate_inactive_examples(
INT *label,              /* Make all variables active again */
double *a, 
SHRINK_STATE *shrink_state,
double *lin,                      /* which had been removed by shrinking. */
LONG totdoc, LONG iteration,      /* Computes lin for those */
LEARN_PARM *learn_parm,              /* variables from scratch. */
LONG *inconsistent,
LONG *docs,
MODEL *model,
REAL *aicache,
double *weights, double* maxdiff)
{
  register LONG i,j,ii,jj,t,*changed2dnum,*inactive2dnum;
  LONG *changed,*inactive;
  register double kernel_val,*a_old,dist;
  double ex_c;

  changed=new long[totdoc];
  changed2dnum=new long[totdoc+11];
  inactive=new long[totdoc];
  inactive2dnum=new long[totdoc+11];
  for(t=shrink_state->deactnum-1;(t>=0) && shrink_state->a_history[t];t--) {
    if(verbosity>=2) {
     CIO::message("%ld..",t);
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
  (*maxdiff)=0;
  for(i=0;i<totdoc;i++) {
    shrink_state->inactive_since[i]=shrink_state->deactnum-1;
    if(!inconsistent[i]) {
      dist=(lin[i]+model->b)*(double)label[i];
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if((a[i]>learn_parm->epsilon_a) && (dist > 1)) {
	if((dist-1.0)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-1.0;
      }
      else if((a[i]<ex_c) && (dist < 1)) {
	if((1.0-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=1.0-dist;
      }
      if((a[i]>(0+learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	shrink_state->active[i]=1;                         /* not at bound */
      }
      else if((a[i]<=(0+learn_parm->epsilon_a)) && (dist < (1+learn_parm->epsilon_shrink))) {
	shrink_state->active[i]=1;
      }
      else if((a[i]>=ex_c)
	      && (dist > (1-learn_parm->epsilon_shrink))) {
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
 delete[] changed;
 delete[] changed2dnum;
 delete[] inactive;
 delete[] inactive2dnum;
} ;


/***********************************************************************/
/*                                                                     */
/*   SVMLight.cpp                                                      */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 19.07.99                                                    */
/*                                                                     */
/*   Copyright (c) 1999  Universitaet Dortmund - All rights reserved   */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/*   THIS INCLUDES THE FOLLOWING ADDITIONS                             */
/*   Generic Kernel Interfacing: Soeren Sonnenburg                     */
/*   Parallizations: Soeren Sonnenburg                                 */
/*   Multiple Kernel Learning: Gunnar Raetsch, Soeren Sonnenburg,      */
/*                          Alexander Zien, Marius Kloft, Chen Guohua  */
/*   Linadd Speedup: Gunnar Raetsch, Soeren Sonnenburg                 */
/*                                                                     */
/***********************************************************************/
#include <shogun/lib/config.h>

#ifdef USE_SVMLIGHT

#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/lapack.h>

#include <shogun/classifier/svm/SVMLight.h>
#include <shogun/lib/external/pr_loqo.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/kernel/CombinedKernel.h>

#include <unistd.h>

#include <shogun/base/Parallel.h>
#include <shogun/labels/BinaryLabels.h>

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct S_THREAD_PARAM_REACTIVATE_LINADD
{
	CKernel* kernel;
	float64_t* lin;
	float64_t* last_lin;
	int32_t* active;
	int32_t* docs;
	int32_t start;
	int32_t end;
};

struct S_THREAD_PARAM_SVMLIGHT
{
	float64_t * lin ;
	float64_t* W;
	int32_t start, end;
	int32_t * active2dnum ;
	int32_t * docs ;
	CKernel* kernel ;
};

struct S_THREAD_PARAM_REACTIVATE_VANILLA
{
	CKernel* kernel;
	float64_t* lin;
	float64_t* aicache;
	float64_t* a;
	float64_t* a_old;
	int32_t* changed2dnum;
	int32_t* inactive2dnum;
	int32_t* label;
	int32_t start;
	int32_t end;
};

struct S_THREAD_PARAM_KERNEL
{
	float64_t *Kval ;
	int32_t *KI, *KJ ;
	int32_t start, end;
    CSVMLight* svmlight;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

void* CSVMLight::update_linear_component_linadd_helper(void* p)
{
	S_THREAD_PARAM_SVMLIGHT* params = (S_THREAD_PARAM_SVMLIGHT*) p;

	int32_t jj=0, j=0 ;

	for (jj=params->start;(jj<params->end) && (j=params->active2dnum[jj])>=0;jj++)
		params->lin[j]+=params->kernel->compute_optimized(params->docs[j]);

	return NULL ;
}

void* CSVMLight::compute_kernel_helper(void* p)
{
	S_THREAD_PARAM_KERNEL* params = (S_THREAD_PARAM_KERNEL*) p;

	int32_t jj=0 ;
	for (jj=params->start;jj<params->end;jj++)
		params->Kval[jj]=params->svmlight->compute_kernel(params->KI[jj], params->KJ[jj]) ;

	return NULL ;
}

CSVMLight::CSVMLight()
: CSVM()
{
	init();
	set_kernel(NULL);
}

CSVMLight::CSVMLight(float64_t C, CKernel* k, CLabels* lab)
: CSVM(C, k, lab)
{
	init();
}

void CSVMLight::init()
{
	//optimizer settings
	primal=NULL;
	dual=NULL;
	init_margin=0.15;
	init_iter=500;
	precision_violations=0;
	model_b=0;
	verbosity=1;
	opt_precision=DEF_PRECISION;

	// svm variables
	W=NULL;
	model=SG_MALLOC(MODEL, 1);
	learn_parm=SG_MALLOC(LEARN_PARM, 1);
	model->supvec=NULL;
	model->alpha=NULL;
	model->index=NULL;

	// MKL stuff
	mymaxdiff=1 ;
	mkl_converged=false;
}

CSVMLight::~CSVMLight()
{

  SG_FREE(model->supvec);
  SG_FREE(model->alpha);
  SG_FREE(model->index);
  SG_FREE(model);
  SG_FREE(learn_parm);

  // MKL stuff
  SG_FREE(W);

  // optimizer variables
  SG_FREE(dual);
  SG_FREE(primal);
}

bool CSVMLight::train_machine(CFeatures* data)
{
	//certain setup params
	mkl_converged=false;
	verbosity=1 ;
	init_margin=0.15;
	init_iter=500;
	precision_violations=0;
	opt_precision=DEF_PRECISION;

	strcpy (learn_parm->predfile, "");
	learn_parm->biased_hyperplane= get_bias_enabled() ? 1 : 0;
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=get_qpsize();
	learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize-1;
	learn_parm->maxiter=100000;
	learn_parm->svm_iter_to_shrink=100;
	learn_parm->svm_c=C1;
	learn_parm->transduction_posratio=0.33;
	learn_parm->svm_costratio=C2/C1;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=epsilon; // GU: better decrease it ... ??
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;

    if (!kernel)
        SG_ERROR("SVM_light can not proceed without kernel!\n")

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

    if (!kernel->has_features())
        SG_ERROR("SVM_light can not proceed without initialized kernel!\n")

	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(m_labels->get_label_type() == LT_BINARY)
	ASSERT(kernel->get_num_vec_lhs()==m_labels->get_num_labels())

	// in case of LINADD enabled kernels cleanup!
	if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
		kernel->clear_normal() ;

	// output some info
	SG_DEBUG("threads = %i\n", parallel->get_num_threads())
	SG_DEBUG("qpsize = %i\n", learn_parm->svm_maxqpsize)
	SG_DEBUG("epsilon = %1.1e\n", learn_parm->epsilon_crit)
	SG_DEBUG("kernel->has_property(KP_LINADD) = %i\n", kernel->has_property(KP_LINADD))
	SG_DEBUG("kernel->has_property(KP_KERNCOMBINATION) = %i\n", kernel->has_property(KP_KERNCOMBINATION))
	SG_DEBUG("kernel->has_property(KP_BATCHEVALUATION) = %i\n", kernel->has_property(KP_BATCHEVALUATION))
	SG_DEBUG("kernel->get_optimization_type() = %s\n", kernel->get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" : "SLOWBUTMEMEFFICIENT" )
	SG_DEBUG("get_solver_type() = %i\n", get_solver_type())
	SG_DEBUG("get_linadd_enabled() = %i\n", get_linadd_enabled())
	SG_DEBUG("get_batch_computation_enabled() = %i\n", get_batch_computation_enabled())
	SG_DEBUG("kernel->get_num_subkernels() = %i\n", kernel->get_num_subkernels())

	use_kernel_cache = !((kernel->get_kernel_type() == K_CUSTOM) ||
						 (get_linadd_enabled() && kernel->has_property(KP_LINADD)));

	SG_DEBUG("use_kernel_cache = %i\n", use_kernel_cache)

	if (kernel->get_kernel_type() == K_COMBINED)
	{

		for (index_t k_idx=0; k_idx<((CCombinedKernel*) kernel)->get_num_kernels(); k_idx++)
		{
			CKernel* kn = ((CCombinedKernel*) kernel)->get_kernel(k_idx);
			// allocate kernel cache but clean up beforehand
			kn->resize_kernel_cache(kn->get_cache_size());
			SG_UNREF(kn);
		}
	}

	kernel->resize_kernel_cache(kernel->get_cache_size());

	// train the svm
	svm_learn();

	// brain damaged svm light work around
	create_new_model(model->sv_num-1);
	set_bias(-model->b);
	for (int32_t i=0; i<model->sv_num-1; i++)
	{
		set_alpha(i, model->alpha[i+1]);
		set_support_vector(i, model->supvec[i+1]);
	}

	// in case of LINADD enabled kernels cleanup!
	if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
	{
		kernel->clear_normal() ;
		kernel->delete_optimization() ;
	}

	if (use_kernel_cache)
		kernel->kernel_cache_cleanup();

	return true ;
}

int32_t CSVMLight::get_runtime()
{
  clock_t start;
  start = clock();
  return((int32_t)((float64_t)start*100.0/(float64_t)CLOCKS_PER_SEC));
}


void CSVMLight::svm_learn()
{
	int32_t *inconsistent, i;
	int32_t misclassified,upsupvecnum;
	float64_t maxdiff, *lin, *c, *a;
	int32_t iterations;
	int32_t trainpos=0, trainneg=0 ;
	ASSERT(m_labels)
	SGVector<int32_t> lab=((CBinaryLabels*) m_labels)->get_int_labels();
	int32_t totdoc=lab.vlen;
	ASSERT(lab.vector && lab.vlen)
	int32_t* label=SGVector<int32_t>::clone_vector(lab.vector, lab.vlen);

	int32_t* docs=SG_MALLOC(int32_t, totdoc);
	SG_FREE(W);
	W=NULL;
	count = 0 ;

	if (kernel->has_property(KP_KERNCOMBINATION) && callback)
	{
		W = SG_MALLOC(float64_t, totdoc*kernel->get_num_subkernels());
		for (i=0; i<totdoc*kernel->get_num_subkernels(); i++)
			W[i]=0;
	}

	for (i=0; i<totdoc; i++)
		docs[i]=i;

	float64_t *xi_fullset; /* buffer for storing xi on full sample in loo */
	float64_t *a_fullset;  /* buffer for storing alpha on full sample in loo */
	TIMING timing_profile;
	SHRINK_STATE shrink_state;

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

	init_shrink_state(&shrink_state,totdoc,(int32_t)MAXSHRINK);

	inconsistent = SG_MALLOC(int32_t, totdoc);
	c = SG_MALLOC(float64_t, totdoc);
	a = SG_MALLOC(float64_t, totdoc);
	a_fullset = SG_MALLOC(float64_t, totdoc);
	xi_fullset = SG_MALLOC(float64_t, totdoc);
	lin = SG_MALLOC(float64_t, totdoc);
	if (m_linear_term.vlen>0)
		learn_parm->eps=get_linear_term_array();
	else
	{
		learn_parm->eps=SG_MALLOC(float64_t, totdoc);      /* equivalent regression epsilon for classification */
		SGVector<float64_t>::fill_vector(learn_parm->eps, totdoc, -1.0);
	}

	learn_parm->svm_cost = SG_MALLOC(float64_t, totdoc);

	SG_FREE(model->supvec);
	SG_FREE(model->alpha);
	SG_FREE(model->index);
	model->supvec = SG_MALLOC(int32_t, totdoc+2);
	model->alpha = SG_MALLOC(float64_t, totdoc+2);
	model->index = SG_MALLOC(int32_t, totdoc+2);

	model->at_upper_bound=0;
	model->b=0;
	model->supvec[0]=0;  /* element 0 reserved and empty for now */
	model->alpha[0]=0;
	model->totdoc=totdoc;

	model->kernel=kernel;

	model->sv_num=1;
	model->loo_error=-1;
	model->loo_recall=-1;
	model->loo_precision=-1;
	model->xa_error=-1;
	model->xa_recall=-1;
	model->xa_precision=-1;

	for (i=0;i<totdoc;i++) {    /* various inits */
		inconsistent[i]=0;
		c[i]=0;
		a[i]=0;
		lin[i]=0;

		if(label[i] > 0) {
			learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
				fabs((float64_t)label[i]);
			label[i]=1;
			trainpos++;
		}
		else if(label[i] < 0) {
			learn_parm->svm_cost[i]=learn_parm->svm_c*fabs((float64_t)label[i]);
			label[i]=-1;
			trainneg++;
		}
		else {
			learn_parm->svm_cost[i]=0;
		}
	}

  /* compute starting state for initial alpha values */
	SG_DEBUG("alpha:%d num_sv:%d\n", m_alpha.vector, get_num_support_vectors())

	if(m_alpha.vector && get_num_support_vectors()) {
		if(verbosity>=1) {
			SG_INFO("Computing starting state...")
		}

	float64_t* alpha = SG_MALLOC(float64_t, totdoc);

	for (i=0; i<totdoc; i++)
		alpha[i]=0;

	for (i=0; i<get_num_support_vectors(); i++)
		alpha[get_support_vector(i)]=get_alpha(i);

    int32_t* index = SG_MALLOC(int32_t, totdoc);
    int32_t* index2dnum = SG_MALLOC(int32_t, totdoc+11);
    float64_t* aicache = SG_MALLOC(float64_t, totdoc);
    for (i=0;i<totdoc;i++) {    /* create full index and clip alphas */
      index[i]=1;
      alpha[i]=fabs(alpha[i]);
      if(alpha[i]<0) alpha[i]=0;
      if(alpha[i]>learn_parm->svm_cost[i]) alpha[i]=learn_parm->svm_cost[i];
    }

	if (use_kernel_cache)
	{
		if (callback &&
				(!((CCombinedKernel*) kernel)->get_append_subkernel_weights())
		   )
		{
			CCombinedKernel* k = (CCombinedKernel*) kernel;
			for (index_t k_idx=0; k_idx<k->get_num_kernels(); k_idx++)
			{
				CKernel* kn = k->get_kernel(k_idx);
				for (i=0;i<totdoc;i++)     // fill kernel cache with unbounded SV
					if((alpha[i]>0) && (alpha[i]<learn_parm->svm_cost[i])
							&& (kn->kernel_cache_space_available()))
						kn->cache_kernel_row(i);

				for (i=0;i<totdoc;i++)     // fill rest of kernel cache with bounded SV
					if((alpha[i]==learn_parm->svm_cost[i])
							&& (kn->kernel_cache_space_available()))
						kn->cache_kernel_row(i);

				SG_UNREF(kn);
			}
		}
		else
		{
			for (i=0;i<totdoc;i++)     /* fill kernel cache with unbounded SV */
				if((alpha[i]>0) && (alpha[i]<learn_parm->svm_cost[i])
						&& (kernel->kernel_cache_space_available()))
					kernel->cache_kernel_row(i);

			for (i=0;i<totdoc;i++)     /* fill rest of kernel cache with bounded SV */
				if((alpha[i]==learn_parm->svm_cost[i])
						&& (kernel->kernel_cache_space_available()))
					kernel->cache_kernel_row(i);
		}
	}
    compute_index(index,totdoc,index2dnum);
    update_linear_component(docs,label,index2dnum,alpha,a,index2dnum,totdoc,
			    lin,aicache,NULL);
    calculate_svm_model(docs,label,lin,alpha,a,c,
			      index2dnum,index2dnum);
    for (i=0;i<totdoc;i++) {    /* copy initial alphas */
      a[i]=alpha[i];
    }

    SG_FREE(index);
    SG_FREE(index2dnum);
    SG_FREE(aicache);
    SG_FREE(alpha);

    if(verbosity>=1)
		SG_DONE()
  }
		SG_DEBUG("%d totdoc %d pos %d neg\n", totdoc, trainpos, trainneg)
		SG_DEBUG("Optimizing...\n")

	/* train the svm */
  iterations=optimize_to_convergence(docs,label,totdoc,
                     &shrink_state,inconsistent,a,lin,
                     c,&timing_profile,
                     &maxdiff,(int32_t)-1,
                     (int32_t)1);


	if(verbosity>=1) {
		if(verbosity==1)
		{
			SG_DONE()
			SG_DEBUG("(%ld iterations)", iterations)
		}

		misclassified=0;
		for (i=0;(i<totdoc);i++) { /* get final statistic */
			if((lin[i]-model->b)*(float64_t)label[i] <= 0.0)
				misclassified++;
		}

		SG_INFO("Optimization finished (%ld misclassified, maxdiff=%.8f).\n",
				misclassified,maxdiff);

		SG_INFO("obj = %.16f, rho = %.16f\n",get_objective(),model->b)
		if (maxdiff>epsilon)
			SG_WARNING("maximum violation (%f) exceeds svm_epsilon (%f) due to numerical difficulties\n", maxdiff, epsilon)

		upsupvecnum=0;
		for (i=1;i<model->sv_num;i++)
		{
			if(fabs(model->alpha[i]) >=
					(learn_parm->svm_cost[model->supvec[i]]-
					 learn_parm->epsilon_a))
				upsupvecnum++;
		}
		SG_INFO("Number of SV: %d (including %d at upper bound)\n",
				model->sv_num-1,upsupvecnum);
	}

	shrink_state_cleanup(&shrink_state);
	SG_FREE(label);
	SG_FREE(inconsistent);
	SG_FREE(c);
	SG_FREE(a);
	SG_FREE(a_fullset);
	SG_FREE(xi_fullset);
	SG_FREE(lin);
	SG_FREE(learn_parm->eps);
	SG_FREE(learn_parm->svm_cost);
	SG_FREE(docs);
}

int32_t CSVMLight::optimize_to_convergence(int32_t* docs, int32_t* label, int32_t totdoc,
			     SHRINK_STATE *shrink_state,
			     int32_t *inconsistent,
			     float64_t *a, float64_t *lin, float64_t *c,
			     TIMING *timing_profile, float64_t *maxdiff,
			     int32_t heldout, int32_t retrain)
     /* docs: Training vectors (x-part) */
     /* label: Training labels/value (y-part, zero if test example for
			      transduction) */
     /* totdoc: Number of examples in docs/label */
     /* laern_parm: Learning paramenters */
     /* kernel_parm: Kernel paramenters */
     /* kernel_cache: Initialized/partly filled Cache, if using a kernel.
                      NULL if linear. */
     /* shrink_state: State of active variables */
     /* inconsistent: examples thrown out as inconstistent */
     /* a: alphas */
     /* lin: linear component of gradient */
     /* c: right hand side of inequalities (margin) */
     /* maxdiff: returns maximum violation of KT-conditions */
     /* heldout: marks held-out example for leave-one-out (or -1) */
     /* retrain: selects training mode (1=regular / 2=holdout) */
{

  int32_t *chosen,*key,i,j,jj,*last_suboptimal_at,noshrink;
  int32_t inconsistentnum,choosenum,already_chosen=0,iteration;
  int32_t misclassified,supvecnum=0,*active2dnum,inactivenum;
  int32_t *working2dnum,*selexam;
  int32_t activenum;
  float64_t criterion, eq;
  float64_t *a_old;
  int32_t t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  float64_t epsilon_crit_org;
  float64_t bestmaxdiff;
  float64_t worstmaxdiff;
  int32_t   bestmaxdiffiter,terminate;
  bool reactivated=false;

  float64_t *selcrit;  /* buffer for sorting */
  float64_t *aicache;  /* buffer to keep one row of hessian */
  QP qp;            /* buffer for one quadratic program */

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
  if(kernel->has_property(KP_LINADD) && get_linadd_enabled()) {
	  learn_parm->epsilon_crit=2.0;
      /* caching makes no sense for linear kernel */
  }
  learn_parm->epsilon_shrink=2;
  (*maxdiff)=1;

  SG_DEBUG("totdoc:%d\n",totdoc)
  chosen = SG_MALLOC(int32_t, totdoc);
  last_suboptimal_at =SG_MALLOC(int32_t, totdoc);
  key =SG_MALLOC(int32_t, totdoc+11);
  selcrit =SG_MALLOC(float64_t, totdoc);
  selexam =SG_MALLOC(int32_t, totdoc);
  a_old =SG_MALLOC(float64_t, totdoc);
  aicache =SG_MALLOC(float64_t, totdoc);
  working2dnum =SG_MALLOC(int32_t, totdoc+11);
  active2dnum =SG_MALLOC(int32_t, totdoc+11);
  qp.opt_ce =SG_MALLOC(float64_t, learn_parm->svm_maxqpsize);
  qp.opt_ce0 =SG_MALLOC(float64_t, 1);
  qp.opt_g =SG_MALLOC(float64_t, learn_parm->svm_maxqpsize*learn_parm->svm_maxqpsize);
  qp.opt_g0 =SG_MALLOC(float64_t, learn_parm->svm_maxqpsize);
  qp.opt_xinit =SG_MALLOC(float64_t, learn_parm->svm_maxqpsize);
  qp.opt_low=SG_MALLOC(float64_t, learn_parm->svm_maxqpsize);
  qp.opt_up=SG_MALLOC(float64_t, learn_parm->svm_maxqpsize);

  choosenum=0;
  inconsistentnum=0;
  if(!retrain) retrain=1;
  iteration=1;
  bestmaxdiffiter=1;
  bestmaxdiff=999999999;
  worstmaxdiff=1e-10;
  terminate=0;


  kernel->set_time(iteration);  /* for lru cache */

  for (i=0;i<totdoc;i++) {    /* various inits */
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
  CTime start_time;
  mkl_converged=false;


#ifdef CYGWIN
  for (;((iteration<100 || (!mkl_converged && callback) ) || (retrain && (!terminate))); iteration++){
#else
	  CSignal::clear_cancel();
	  for (;((!CSignal::cancel_computations()) && ((iteration<3 || (!mkl_converged && callback) ) || (retrain && (!terminate)))); iteration++){
#endif

	  if(use_kernel_cache)
		  kernel->set_time(iteration);  /* for lru cache */

	  if(verbosity>=2) t0=get_runtime();
	  if(verbosity>=3) {
		  SG_DEBUG("\nSelecting working set... ")
	  }

	  if(learn_parm->svm_newvarsinqp>learn_parm->svm_maxqpsize)
		  learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;

	  i=0;
	  for (jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear working set */
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
		  for (jj=0;(j=working2dnum[jj])>=0;jj++) { /* fully clear working set */
			  chosen[j]=0;
		  }
		  clear_index(working2dnum);
		  for (i=0;i<totdoc;i++) { /* set inconsistent examples to zero (-i 1) */
			  if((inconsistent[i] || (heldout==i)) && (a[i] != 0.0)) {
				  chosen[i]=99999;
				  choosenum++;
				  a[i]=0;
			  }
		  }
		  if(learn_parm->biased_hyperplane) {
			  eq=0;
			  for (i=0;i<totdoc;i++) { /* make sure we fulfill equality constraint */
				  eq+=a[i]*label[i];
			  }
			  for (i=0;(i<totdoc) && (fabs(eq) > learn_parm->epsilon_a);i++) {
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
			  if(CMath::min(learn_parm->svm_newvarsinqp, learn_parm->svm_maxqpsize-choosenum)>=4 &&
					  (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())))
			  {
				  /* select part of the working set from cache */
				  already_chosen=select_next_qp_subproblem_grad(
					  label,a,lin,c,totdoc,
					  (int32_t)(CMath::min(learn_parm->svm_maxqpsize-choosenum,
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
		  SG_INFO(" %ld vectors chosen\n",choosenum)
	  }

	  if(verbosity>=2) t1=get_runtime();

	  if (use_kernel_cache)
	  {
		  // in case of MKL w/o linadd cache each kernel independently
		  // else if linadd is disabled cache single kernel
		  if ( callback &&
				  (!((CCombinedKernel*) kernel)->get_append_subkernel_weights())
			 )
		  {
			  CCombinedKernel* k = (CCombinedKernel*) kernel;
			  for (index_t k_idx=0; k_idx<k->get_num_kernels(); k_idx++)
			  {
				  CKernel* kn = k->get_kernel(k_idx);
				  kn->cache_multiple_kernel_rows(working2dnum, choosenum);
				  SG_UNREF(kn);
			  }
		  }
		  else
			  kernel->cache_multiple_kernel_rows(working2dnum, choosenum);
	  }

	  if(verbosity>=2) t2=get_runtime();

	  if(retrain != 2) {
		  optimize_svm(docs,label,inconsistent,0.0,chosen,active2dnum,
					   totdoc,working2dnum,choosenum,a,lin,c,
					   aicache,&qp,&epsilon_crit_org);
	  }

	  if(verbosity>=2) t3=get_runtime();
	  update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
							  lin,aicache,c);

	  if(verbosity>=2) t4=get_runtime();
	  supvecnum=calculate_svm_model(docs,label,lin,a,a_old,c,working2dnum,active2dnum);

	  if(verbosity>=2) t5=get_runtime();

	  for (jj=0;(i=working2dnum[jj])>=0;jj++) {
		  a_old[i]=a[i];
	  }

	  retrain=check_optimality(label,a,lin,c,totdoc,
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
		  /* int32_t time no progress? */
		  terminate=1;
		  retrain=0;
		  SG_WARNING("Relaxing KT-Conditions due to slow progress! Terminating!\n")
		  SG_DEBUG("(iteration :%d, bestmaxdiffiter: %d, lean_param->maxiter: %d)\n", iteration, bestmaxdiffiter, learn_parm->maxiter )
	  }

	  noshrink= (get_shrinking_enabled()) ? 0 : 1;

	  if ((!callback) && (!retrain) && (inactivenum>0) &&
			  ((!learn_parm->skip_final_opt_check) || (kernel->has_property(KP_LINADD) && get_linadd_enabled())))
	  {
		  t1=get_runtime();
		  SG_DEBUG("reactivating inactive examples\n")

		  reactivate_inactive_examples(label,a,shrink_state,lin,c,totdoc,
									   iteration,inconsistent,
									   docs,aicache,
									   maxdiff);
		  reactivated=true;
		  SG_DEBUG("done reactivating inactive examples (maxdiff:%8f eps_crit:%8f orig_eps:%8f)\n", *maxdiff, learn_parm->epsilon_crit, epsilon_crit_org)
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
		  {
			  SG_INFO("restarting optimization as we are - due to shrinking - deviating too much (maxdiff(%f) > eps(%f))\n", *maxdiff, learn_parm->epsilon_crit)
		      retrain=1;
		  }
		  timing_profile->time_shrink+=get_runtime()-t1;
		  if (((verbosity>=1) && (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())))
		     || (verbosity>=2)) {
		      SG_DONE()
		      SG_DEBUG("Number of inactive variables = %ld\n", inactivenum)
		  }
	  }

	  if((!retrain) && (learn_parm->epsilon_crit>(*maxdiff)))
		  learn_parm->epsilon_crit=(*maxdiff);
	  if((!retrain) && (learn_parm->epsilon_crit>epsilon_crit_org)) {
		  learn_parm->epsilon_crit/=4.0;
		  retrain=1;
		  noshrink=1;
	  }
	  if(learn_parm->epsilon_crit<epsilon_crit_org)
		  learn_parm->epsilon_crit=epsilon_crit_org;

	  if(verbosity>=2) {
		  SG_INFO(" => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
					   supvecnum,model->at_upper_bound,(*maxdiff));

	  }
	  mymaxdiff=*maxdiff ;

	  //don't shrink w/ mkl
	  if (((iteration % 10) == 0) && (!noshrink) && !callback)
	  {
		  activenum=shrink_problem(shrink_state,active2dnum,last_suboptimal_at,iteration,totdoc,
				  CMath::max((int32_t)(activenum/10),
					  CMath::max((int32_t)(totdoc/500),(int32_t) 100)),
				  a,inconsistent, c, lin, label);

		  inactivenum=totdoc-activenum;

		  if (use_kernel_cache && (supvecnum>kernel->get_max_elems_cache())
				  && ((kernel->get_activenum_cache()-activenum)>CMath::max((int32_t)(activenum/10),(int32_t) 500))) {

			  kernel->kernel_cache_shrink(totdoc, CMath::min((int32_t) (kernel->get_activenum_cache()-activenum),
						  (int32_t) (kernel->get_activenum_cache()-supvecnum)),
					  shrink_state->active);
		  }
	  }

	  if (bestmaxdiff>worstmaxdiff)
		  worstmaxdiff=bestmaxdiff;

	  SG_ABS_PROGRESS(bestmaxdiff, -CMath::log10(bestmaxdiff), -CMath::log10(worstmaxdiff), -CMath::log10(epsilon), 6)

	  /* Terminate loop */
	  if (m_max_train_time > 0 && start_time.cur_time_diff() > m_max_train_time) {
	      terminate = 1;
	      retrain = 0;
	  }

  } /* end of loop */

  SG_DEBUG("inactive:%d\n", inactivenum)

  if (inactivenum && !reactivated && !callback)
  {
      SG_DEBUG("reactivating inactive examples\n")
      reactivate_inactive_examples(label,a,shrink_state,lin,c,totdoc,
		  iteration,inconsistent,
		  docs,aicache,
		  maxdiff);
      SG_DEBUG("done reactivating inactive examples\n")
      /* Update to new active variables. */
      activenum=compute_index(shrink_state->active,totdoc,active2dnum);
      inactivenum=totdoc-activenum;
      /* reset watchdog */
      bestmaxdiff=(*maxdiff);
      bestmaxdiffiter=iteration;
  }

  //use this for our purposes!
  criterion=compute_objective_function(a,lin,c,learn_parm->eps,label,totdoc);
  CSVM::set_objective(criterion);

  SG_FREE(chosen);
  SG_FREE(last_suboptimal_at);
  SG_FREE(key);
  SG_FREE(selcrit);
  SG_FREE(selexam);
  SG_FREE(a_old);
  SG_FREE(aicache);
  SG_FREE(working2dnum);
  SG_FREE(active2dnum);
  SG_FREE(qp.opt_ce);
  SG_FREE(qp.opt_ce0);
  SG_FREE(qp.opt_g);
  SG_FREE(qp.opt_g0);
  SG_FREE(qp.opt_xinit);
  SG_FREE(qp.opt_low);
  SG_FREE(qp.opt_up);

  learn_parm->epsilon_crit=epsilon_crit_org; /* restore org */

  return(iteration);
}

float64_t CSVMLight::compute_objective_function(
	float64_t *a, float64_t *lin, float64_t *c, float64_t* eps, int32_t *label,
	int32_t totdoc)
     /* Return value of objective function. */
     /* Works only relative to the active variables! */
{
  /* calculate value of objective function */
  float64_t criterion=0;

  for (int32_t i=0;i<totdoc;i++)
	  criterion=criterion+(eps[i]-(float64_t)label[i]*c[i])*a[i]+0.5*a[i]*label[i]*lin[i];

  return(criterion);
}


void CSVMLight::clear_index(int32_t *index)
              /* initializes and empties index */
{
  index[0]=-1;
}

void CSVMLight::add_to_index(int32_t *index, int32_t elem)
     /* initializes and empties index */
{
  register int32_t i;
  for (i=0;index[i] != -1;i++);
  index[i]=elem;
  index[i+1]=-1;
}

int32_t CSVMLight::compute_index(
	int32_t *binfeature, int32_t range, int32_t *index)
     /* create an inverted index of binfeature */
{
  register int32_t i,ii;

  ii=0;
  for (i=0;i<range;i++) {
    if(binfeature[i]) {
      index[ii]=i;
      ii++;
    }
  }
  for (i=0;i<4;i++) {
    index[ii+i]=-1;
  }
  return(ii);
}


void CSVMLight::optimize_svm(
	int32_t* docs, int32_t* label, int32_t *exclude_from_eq_const,
	float64_t eq_target, int32_t *chosen, int32_t *active2dnum, int32_t totdoc,
	int32_t *working2dnum, int32_t varnum, float64_t *a, float64_t *lin,
	float64_t *c, float64_t *aicache, QP *qp, float64_t *epsilon_crit_target)
     /* Do optimization on the working set. */
{
    int32_t i;
    float64_t *a_v;

    //compute_matrices_for_optimization_parallel(docs,label,
	//										   exclude_from_eq_const,eq_target,chosen,
	//										   active2dnum,working2dnum,a,lin,c,
	//										   varnum,totdoc,aicache,qp);

    compute_matrices_for_optimization(docs,label,
									  exclude_from_eq_const,eq_target,chosen,
									  active2dnum,working2dnum,a,lin,c,
									  varnum,totdoc,aicache,qp);

    if(verbosity>=3) {
     SG_DEBUG("Running optimizer...")
    }
    /* call the qp-subsolver */
    a_v=optimize_qp(qp,epsilon_crit_target,
		    learn_parm->svm_maxqpsize,
		    &(model->b),				/* in case the optimizer gives us */
            learn_parm->svm_maxqpsize); /* the threshold for free. otherwise */
		/* b is calculated in calculate_model. */
    if(verbosity>=3) {
     SG_DONE()
    }

    for (i=0;i<varnum;i++)
      a[working2dnum[i]]=a_v[i];
}

void CSVMLight::compute_matrices_for_optimization_parallel(
	int32_t* docs, int32_t* label, int32_t *exclude_from_eq_const,
	float64_t eq_target, int32_t *chosen, int32_t *active2dnum, int32_t *key,
	float64_t *a, float64_t *lin, float64_t *c, int32_t varnum, int32_t totdoc,
	float64_t *aicache, QP *qp)
{
	if (parallel->get_num_threads()<=1)
	{
		compute_matrices_for_optimization(docs, label, exclude_from_eq_const, eq_target,
												   chosen, active2dnum, key, a, lin, c,
												   varnum, totdoc, aicache, qp) ;
	}
#ifdef HAVE_PTHREAD
	else
	{
		register int32_t ki,kj,i,j;
		register float64_t kernel_temp;

		qp->opt_n=varnum;
		qp->opt_ce0[0]=-eq_target; /* compute the constant for equality constraint */
		for (j=1;j<model->sv_num;j++) { /* start at 1 */
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
		for (i=0;i<varnum;i++) {
			qp->opt_g0[i]=lin[key[i]];
		}

		ASSERT(parallel->get_num_threads()>1)
		int32_t *KI=SG_MALLOC(int32_t, varnum*varnum);
		int32_t *KJ=SG_MALLOC(int32_t, varnum*varnum);
		int32_t Knum=0 ;
		float64_t *Kval = SG_MALLOC(float64_t, varnum*(varnum+1)/2);
		for (i=0;i<varnum;i++) {
			ki=key[i];
			KI[Knum]=docs[ki] ;
			KJ[Knum]=docs[ki] ;
			Knum++ ;
			for (j=i+1;j<varnum;j++)
			{
				kj=key[j];
				KI[Knum]=docs[ki] ;
				KJ[Knum]=docs[kj] ;
				Knum++ ;
			}
		}
		ASSERT(Knum<=varnum*(varnum+1)/2)

		pthread_t* threads = SG_MALLOC(pthread_t, parallel->get_num_threads()-1);
		S_THREAD_PARAM_KERNEL* params = SG_MALLOC(S_THREAD_PARAM_KERNEL, parallel->get_num_threads()-1);
		int32_t step= Knum/parallel->get_num_threads();
		//SG_DEBUG("\nkernel-step size: %i\n", step)
		for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
		{
			params[t].svmlight = this;
			params[t].start = t*step;
			params[t].end = (t+1)*step;
			params[t].KI=KI ;
			params[t].KJ=KJ ;
			params[t].Kval=Kval ;
			pthread_create(&threads[t], NULL, CSVMLight::compute_kernel_helper, (void*)&params[t]);
		}
		for (i=params[parallel->get_num_threads()-2].end; i<Knum; i++)
			Kval[i]=compute_kernel(KI[i],KJ[i]) ;

		for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
			pthread_join(threads[t], NULL);

		SG_FREE(params);
		SG_FREE(threads);

		Knum=0 ;
		for (i=0;i<varnum;i++) {
			ki=key[i];

			/* Compute the matrix for equality constraints */
			qp->opt_ce[i]=label[ki];
			qp->opt_low[i]=0;
			qp->opt_up[i]=learn_parm->svm_cost[ki];

			kernel_temp=Kval[Knum] ;
			Knum++ ;
			/* compute linear part of objective function */
			qp->opt_g0[i]-=(kernel_temp*a[ki]*(float64_t)label[ki]);
			/* compute quadratic part of objective function */
			qp->opt_g[varnum*i+i]=kernel_temp;

			for (j=i+1;j<varnum;j++) {
				kj=key[j];
				kernel_temp=Kval[Knum] ;
				Knum++ ;
				/* compute linear part of objective function */
				qp->opt_g0[i]-=(kernel_temp*a[kj]*(float64_t)label[kj]);
				qp->opt_g0[j]-=(kernel_temp*a[ki]*(float64_t)label[ki]);
				/* compute quadratic part of objective function */
				qp->opt_g[varnum*i+j]=(float64_t)label[ki]*(float64_t)label[kj]*kernel_temp;
				qp->opt_g[varnum*j+i]=qp->opt_g[varnum*i+j];//(float64_t)label[ki]*(float64_t)label[kj]*kernel_temp;
			}

			if(verbosity>=3) {
				if(i % 20 == 0) {
					SG_DEBUG("%ld..",i)
				}
			}
		}

		SG_FREE(KI);
		SG_FREE(KJ);
		SG_FREE(Kval);

		for (i=0;i<varnum;i++) {
			/* assure starting at feasible point */
			qp->opt_xinit[i]=a[key[i]];
			/* set linear part of objective function */
			qp->opt_g0[i]=(learn_parm->eps[key[i]]-(float64_t)label[key[i]]*c[key[i]])+qp->opt_g0[i]*(float64_t)label[key[i]];
		}

		if(verbosity>=3) {
			SG_DONE()
		}
	}
#endif
}

void CSVMLight::compute_matrices_for_optimization(
	int32_t* docs, int32_t* label, int32_t *exclude_from_eq_const,
	float64_t eq_target, int32_t *chosen, int32_t *active2dnum, int32_t *key,
	float64_t *a, float64_t *lin, float64_t *c, int32_t varnum, int32_t totdoc,
	float64_t *aicache, QP *qp)
{
  register int32_t ki,kj,i,j;
  register float64_t kernel_temp;

  qp->opt_n=varnum;
  qp->opt_ce0[0]=-eq_target; /* compute the constant for equality constraint */
  for (j=1;j<model->sv_num;j++) { /* start at 1 */
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
  for (i=0;i<varnum;i++) {
    qp->opt_g0[i]=lin[key[i]];
  }

  for (i=0;i<varnum;i++) {
	  ki=key[i];

	  /* Compute the matrix for equality constraints */
	  qp->opt_ce[i]=label[ki];
	  qp->opt_low[i]=0;
	  qp->opt_up[i]=learn_parm->svm_cost[ki];

	  kernel_temp=compute_kernel(docs[ki], docs[ki]);
	  /* compute linear part of objective function */
	  qp->opt_g0[i]-=(kernel_temp*a[ki]*(float64_t)label[ki]);
	  /* compute quadratic part of objective function */
	  qp->opt_g[varnum*i+i]=kernel_temp;

	  for (j=i+1;j<varnum;j++) {
		  kj=key[j];
		  kernel_temp=compute_kernel(docs[ki], docs[kj]);

		  /* compute linear part of objective function */
		  qp->opt_g0[i]-=(kernel_temp*a[kj]*(float64_t)label[kj]);
		  qp->opt_g0[j]-=(kernel_temp*a[ki]*(float64_t)label[ki]);
		  /* compute quadratic part of objective function */
		  qp->opt_g[varnum*i+j]=(float64_t)label[ki]*(float64_t)label[kj]*kernel_temp;
		  qp->opt_g[varnum*j+i]=qp->opt_g[varnum*i+j];//(float64_t)label[ki]*(float64_t)label[kj]*kernel_temp;
	  }

	  if(verbosity>=3) {
		  if(i % 20 == 0) {
			  SG_DEBUG("%ld..",i)
		  }
	  }
  }

  for (i=0;i<varnum;i++) {
	  /* assure starting at feasible point */
	  qp->opt_xinit[i]=a[key[i]];
	  /* set linear part of objective function */
	  qp->opt_g0[i]=(learn_parm->eps[key[i]]-(float64_t)label[key[i]]*c[key[i]]) + qp->opt_g0[i]*(float64_t)label[key[i]];
  }

  if(verbosity>=3) {
	  SG_DONE()
  }
}


int32_t CSVMLight::calculate_svm_model(
	int32_t* docs, int32_t *label, float64_t *lin, float64_t *a,
	float64_t *a_old, float64_t *c, int32_t *working2dnum, int32_t *active2dnum)
     /* Compute decision function based on current values */
     /* of alpha. */
{
  int32_t i,ii,pos,b_calculated=0,first_low,first_high;
  float64_t ex_c,b_temp,b_low,b_high;

  if(verbosity>=3) {
   SG_DEBUG("Calculating model...")
  }

  if(!learn_parm->biased_hyperplane) {
    model->b=0;
    b_calculated=1;
  }

  for (ii=0;(i=working2dnum[ii])>=0;ii++) {
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
      model->alpha[model->sv_num]=a[i]*(float64_t)label[i];
      model->index[i]=model->sv_num;
      (model->sv_num)++;
    }
    else if(a_old[i]==a[i]) { /* nothing to do */
    }
    else {  /* just update alpha */
      model->alpha[model->index[i]]=a[i]*(float64_t)label[i];
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
	model->b=((float64_t)label[i]*learn_parm->eps[i]-c[i]+lin[i]);
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
    for (ii=0;(i=active2dnum[ii])>=0;ii++) {
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if(a[i]<ex_c) {
	if(label[i]>0)  {
	  b_temp=-(learn_parm->eps[i]-c[i]+lin[i]);
	  if((b_temp>b_low) || (first_low)) {
	    b_low=b_temp;
	    first_low=0;
	  }
	}
	else {
	  b_temp=-(-learn_parm->eps[i]-c[i]+lin[i]);
	  if((b_temp<b_high) || (first_high)) {
	    b_high=b_temp;
	    first_high=0;
	  }
	}
      }
      else {
	if(label[i]<0)  {
	  b_temp=-(-learn_parm->eps[i]-c[i]+lin[i]);
	  if((b_temp>b_low) || (first_low)) {
	    b_low=b_temp;
	    first_low=0;
	  }
	}
	else {
	  b_temp=-(learn_parm->eps[i]-c[i]+lin[i]);
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
    }
  }

  if(verbosity>=3) {
   SG_DONE()
  }

  return(model->sv_num-1); /* have to substract one, since element 0 is empty*/
}

int32_t CSVMLight::check_optimality(
	int32_t* label, float64_t *a, float64_t *lin, float64_t *c, int32_t totdoc,
	float64_t *maxdiff, float64_t epsilon_crit_org, int32_t *misclassified,
	int32_t *inconsistent, int32_t *active2dnum, int32_t *last_suboptimal_at,
	int32_t iteration)
     /* Check KT-conditions */
{
  int32_t i,ii,retrain;
  float64_t dist,ex_c,target;

  if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
	  learn_parm->epsilon_shrink=-learn_parm->epsilon_crit+epsilon_crit_org;
  else
	  learn_parm->epsilon_shrink=learn_parm->epsilon_shrink*0.7+(*maxdiff)*0.3;
  retrain=0;
  (*maxdiff)=0;
  (*misclassified)=0;
  for (ii=0;(i=active2dnum[ii])>=0;ii++) {
	  if((!inconsistent[i]) && label[i]) {
		  dist=(lin[i]-model->b)*(float64_t)label[i];/* 'distance' from
													 hyperplane*/
		  target=-(learn_parm->eps[i]-(float64_t)label[i]*c[i]);
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
		  /* Count how int32_t a variable was at lower/upper bound (and optimal).*/
		  /* Variables, which were at the bound and optimal for a int32_t */
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

void CSVMLight::update_linear_component(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache, float64_t* c)
     /* keep track of the linear component */
     /* lin of the gradient etc. by updating */
     /* based on the change of the variables */
     /* in the current working set */
{
	register int32_t i=0,ii=0,j=0,jj=0;

	if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
	{
		if (callback)
		{
			update_linear_component_mkl_linadd(docs, label, active2dnum, a,
					a_old, working2dnum, totdoc, lin, aicache);
		}
		else
		{
			kernel->clear_normal();

			int32_t num_working=0;
			for (ii=0;(i=working2dnum[ii])>=0;ii++) {
				if(a[i] != a_old[i]) {
					kernel->add_to_normal(docs[i], (a[i]-a_old[i])*(float64_t)label[i]);
					num_working++;
				}
			}

			if (num_working>0)
			{
				if (parallel->get_num_threads() < 2)
				{
					for (jj=0;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=kernel->compute_optimized(docs[j]);
					}
				}
#ifdef HAVE_PTHREAD
				else
				{
					int32_t num_elem = 0 ;
					for (jj=0;(j=active2dnum[jj])>=0;jj++) num_elem++ ;

					pthread_t* threads = SG_MALLOC(pthread_t, parallel->get_num_threads()-1);
					S_THREAD_PARAM_SVMLIGHT* params = SG_MALLOC(S_THREAD_PARAM_SVMLIGHT, parallel->get_num_threads()-1);
					int32_t start = 0 ;
					int32_t step = num_elem/parallel->get_num_threads();
					int32_t end = step ;

					for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
					{
						params[t].kernel = kernel ;
						params[t].lin = lin ;
						params[t].docs = docs ;
						params[t].active2dnum=active2dnum ;
						params[t].start = start ;
						params[t].end = end ;
						start=end ;
						end+=step ;
						pthread_create(&threads[t], NULL, update_linear_component_linadd_helper, (void*)&params[t]) ;
					}

					for (jj=params[parallel->get_num_threads()-2].end;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=kernel->compute_optimized(docs[j]);
					}
					void* ret;
					for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
						pthread_join(threads[t], &ret) ;

					SG_FREE(params);
					SG_FREE(threads);
				}
#endif
			}
		}
	}
	else
	{
		if (callback)
		{
			update_linear_component_mkl(docs, label, active2dnum,
					a, a_old, working2dnum, totdoc,	lin, aicache);
		}
		else {
			for (jj=0;(i=working2dnum[jj])>=0;jj++) {
				if(a[i] != a_old[i]) {
					kernel->get_kernel_row(i,active2dnum,aicache);
					for (ii=0;(j=active2dnum[ii])>=0;ii++)
						lin[j]+=(a[i]-a_old[i])*aicache[j]*(float64_t)label[i];
				}
			}
		}
	}
}


void CSVMLight::update_linear_component_mkl(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache)
{
	//int inner_iters=0;
	int32_t num = kernel->get_num_vec_rhs();
	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	const float64_t* old_beta   = kernel->get_subkernel_weights(num_weights);
	ASSERT(num_weights==num_kernels)

	if ((kernel->get_kernel_type()==K_COMBINED) &&
			 (!((CCombinedKernel*) kernel)->get_append_subkernel_weights()))// for combined kernel
	{
		CCombinedKernel* k = (CCombinedKernel*) kernel;

		int32_t n = 0, i, j ;

		for (index_t k_idx=0; k_idx<k->get_num_kernels(); k_idx++)
		{
			CKernel* kn = k->get_kernel(k_idx);
			for (i=0;i<num;i++)
			{
				if(a[i] != a_old[i])
				{
					kn->get_kernel_row(i,NULL,aicache, true);
					for (j=0;j<num;j++)
						W[j*num_kernels+n]+=(a[i]-a_old[i])*aicache[j]*(float64_t)label[i];
				}
			}

			SG_UNREF(kn);
			n++ ;
		}
	}
	else // hope the kernel is fast ...
	{
		float64_t* w_backup = SG_MALLOC(float64_t, num_kernels);
		float64_t* w1 = SG_MALLOC(float64_t, num_kernels);

		// backup and set to zero
		for (int32_t i=0; i<num_kernels; i++)
		{
			w_backup[i] = old_beta[i] ;
			w1[i]=0.0 ;
		}
		for (int32_t n=0; n<num_kernels; n++)
		{
			w1[n]=1.0 ;
			kernel->set_subkernel_weights(SGVector<float64_t>(w1, num_weights));

			for (int32_t i=0;i<num;i++)
			{
				if(a[i] != a_old[i])
				{
					for (int32_t j=0;j<num;j++)
						W[j*num_kernels+n]+=(a[i]-a_old[i])*compute_kernel(i,j)*(float64_t)label[i];
				}
			}
			w1[n]=0.0 ;
		}

		// restore old weights
		kernel->set_subkernel_weights(SGVector<float64_t>(w_backup,num_weights));

		SG_FREE(w_backup);
		SG_FREE(w1);
	}

	call_mkl_callback(a, label, lin);
}


void CSVMLight::update_linear_component_mkl_linadd(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache)
{
	//int inner_iters=0;

	// kernel with LP_LINADD property is assumed to have
	// compute_by_subkernel functions
	int32_t num = kernel->get_num_vec_rhs();
	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	const float64_t* old_beta   = kernel->get_subkernel_weights(num_weights);
	ASSERT(num_weights==num_kernels)

	float64_t* w_backup = SG_MALLOC(float64_t, num_kernels);
	float64_t* w1 = SG_MALLOC(float64_t, num_kernels);

	// backup and set to one
	for (int32_t i=0; i<num_kernels; i++)
	{
		w_backup[i] = old_beta[i] ;
		w1[i]=1.0 ;
	}
	// set the kernel weights
	kernel->set_subkernel_weights(SGVector<float64_t>(w1, num_weights));

	// create normal update (with changed alphas only)
	kernel->clear_normal();
	for (int32_t ii=0, i=0;(i=working2dnum[ii])>=0;ii++) {
		if(a[i] != a_old[i]) {
			kernel->add_to_normal(docs[i], (a[i]-a_old[i])*(float64_t)label[i]);
		}
	}

	if (parallel->get_num_threads() < 2)
	{
		// determine contributions of different kernels
		for (int32_t i=0; i<num; i++)
			kernel->compute_by_subkernel(i,&W[i*num_kernels]);
	}
#ifdef HAVE_PTHREAD
	else
	{
		pthread_t* threads = SG_MALLOC(pthread_t, parallel->get_num_threads()-1);
		S_THREAD_PARAM_SVMLIGHT* params = SG_MALLOC(S_THREAD_PARAM_SVMLIGHT, parallel->get_num_threads()-1);
		int32_t step= num/parallel->get_num_threads();

		for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
		{
			params[t].kernel = kernel;
			params[t].W = W;
			params[t].start = t*step;
			params[t].end = (t+1)*step;
			pthread_create(&threads[t], NULL, CSVMLight::update_linear_component_mkl_linadd_helper, (void*)&params[t]);
		}

		for (int32_t i=params[parallel->get_num_threads()-2].end; i<num; i++)
			kernel->compute_by_subkernel(i,&W[i*num_kernels]);

		for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
			pthread_join(threads[t], NULL);

		SG_FREE(params);
		SG_FREE(threads);
	}
#endif

	// restore old weights
	kernel->set_subkernel_weights(SGVector<float64_t>(w_backup,num_weights));

	call_mkl_callback(a, label, lin);
}

void* CSVMLight::update_linear_component_mkl_linadd_helper(void* p)
{
	S_THREAD_PARAM_SVMLIGHT* params = (S_THREAD_PARAM_SVMLIGHT*) p;

	int32_t num_kernels=params->kernel->get_num_subkernels();

	// determine contributions of different kernels
	for (int32_t i=params->start; i<params->end; i++)
		params->kernel->compute_by_subkernel(i,&(params->W[i*num_kernels]));

	return NULL ;
}

void CSVMLight::call_mkl_callback(float64_t* a, int32_t* label, float64_t* lin)
{
	int32_t num = kernel->get_num_vec_rhs();
	int32_t num_kernels = kernel->get_num_subkernels() ;

	float64_t suma=0;
	float64_t* sumw=SG_MALLOC(float64_t, num_kernels);
#ifdef HAVE_LAPACK
    int nk = (int) num_kernels; /* calling external lib */
	double* alphay  = SG_MALLOC(double, num);

	for (int32_t i=0; i<num; i++)
	{
		alphay[i]=a[i]*label[i];
		suma+=a[i];
	}

	for (int32_t i=0; i<num_kernels; i++)
		sumw[i]=0;

	cblas_dgemv(CblasColMajor, CblasNoTrans, num_kernels, (int) num, 0.5, (double*) W,
			num_kernels, alphay, 1, 1.0, (double*) sumw, 1);

	SG_FREE(alphay);
#else
	for (int32_t i=0; i<num; i++)
		suma += a[i];

	for (int32_t d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for (int32_t i=0; i<num; i++)
			sumw[d] += a[i]*(0.5*label[i]*W[i*num_kernels+d]);
	}
#endif

	if (callback)
		mkl_converged=callback(mkl, sumw, suma);


	const float64_t* new_beta   = kernel->get_subkernel_weights(num_kernels);

    // update lin
#ifdef HAVE_LAPACK
    cblas_dgemv(CblasColMajor, CblasTrans, nk, (int) num, 1.0, (double*) W,
        nk, (double*) new_beta, 1, 0.0, (double*) lin, 1);
#else
    for (int32_t i=0; i<num; i++)
        lin[i]=0 ;
    for (int32_t d=0; d<num_kernels; d++)
        if (new_beta[d]!=0)
            for (int32_t i=0; i<num; i++)
                lin[i] += new_beta[d]*W[i*num_kernels+d] ;
#endif

	SG_FREE(sumw);
}


/*************************** Working set selection ***************************/

int32_t CSVMLight::select_next_qp_subproblem_grad(
	int32_t* label, float64_t *a, float64_t *lin, float64_t *c, int32_t totdoc,
	int32_t qp_size, int32_t *inconsistent, int32_t *active2dnum,
	int32_t *working2dnum, float64_t *selcrit, int32_t *select,
	int32_t cache_only, int32_t *key, int32_t *chosen)
	/* Use the feasible direction approach to select the next
	   qp-subproblem (see chapter 'Selecting a good working set'). If
	   'cache_only' is true, then the variables are selected only among
	   those for which the kernel evaluations are cached. */
{
	int32_t choosenum,i,j,k,activedoc,inum,valid;
	float64_t s;

	for (inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
	choosenum=0;
	activedoc=0;
	for (i=0;(j=active2dnum[i])>=0;i++) {
		s=-label[j];
		if(cache_only)
		{
			if (use_kernel_cache)
				valid=(kernel->kernel_cache_check(j));
			else
				valid = 1 ;
		}
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
			selcrit[activedoc]=(float64_t)label[j]*(learn_parm->eps[j]-(float64_t)label[j]*c[j]+(float64_t)label[j]*lin[j]);
			key[activedoc]=j;
			activedoc++;
		}
	}
	select_top_n(selcrit,activedoc,select,(int32_t)(qp_size/2));
	for (k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
		i=key[select[k]];
		chosen[i]=1;
		working2dnum[inum+choosenum]=i;
		choosenum+=1;
		if (use_kernel_cache)
			kernel->kernel_cache_touch(i);
        /* make sure it does not get kicked */
		/* out of cache */
	}

	activedoc=0;
	for (i=0;(j=active2dnum[i])>=0;i++) {
		s=label[j];
		if(cache_only)
		{
			if (use_kernel_cache)
				valid=(kernel->kernel_cache_check(j));
			else
				valid = 1 ;
		}
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
			selcrit[activedoc]=-(float64_t)label[j]*(learn_parm->eps[j]-(float64_t)label[j]*c[j]+(float64_t)label[j]*lin[j]);
			/*  selcrit[activedoc]=-(float64_t)(label[j]*(-1.0+(float64_t)label[j]*lin[j])); */
			key[activedoc]=j;
			activedoc++;
		}
	}
	select_top_n(selcrit,activedoc,select,(int32_t)(qp_size/2));
	for (k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
		i=key[select[k]];
		chosen[i]=1;
		working2dnum[inum+choosenum]=i;
		choosenum+=1;
		if (use_kernel_cache)
			kernel->kernel_cache_touch(i); /* make sure it does not get kicked */
		/* out of cache */
	}
	working2dnum[inum+choosenum]=-1; /* complete index */
	return(choosenum);
}

int32_t CSVMLight::select_next_qp_subproblem_rand(
	int32_t* label, float64_t *a, float64_t *lin, float64_t *c, int32_t totdoc,
	int32_t qp_size, int32_t *inconsistent, int32_t *active2dnum,
	int32_t *working2dnum, float64_t *selcrit, int32_t *select, int32_t *key,
	int32_t *chosen, int32_t iteration)
/* Use the feasible direction approach to select the next
   qp-subproblem (see section 'Selecting a good working set'). Chooses
   a feasible direction at (pseudo) random to help jump over numerical
   problem. */
{
  int32_t choosenum,i,j,k,activedoc,inum;
  float64_t s;

  for (inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for (i=0;(j=active2dnum[i])>=0;i++) {
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
  select_top_n(selcrit,activedoc,select,(int32_t)(qp_size/2));
  for (k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
	if (use_kernel_cache)
		kernel->kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for (i=0;(j=active2dnum[i])>=0;i++) {
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
  select_top_n(selcrit,activedoc,select,(int32_t)(qp_size/2));
  for (k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
	if (use_kernel_cache)
		kernel->kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  }
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}



void CSVMLight::select_top_n(
	float64_t *selcrit, int32_t range, int32_t* select, int32_t n)
{
  register int32_t i,j;

  for (i=0;(i<n) && (i<range);i++) { /* Initialize with the first n elements */
    for (j=i;j>=0;j--) {
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
    for (i=n;i<range;i++) {
      if(selcrit[i]>selcrit[select[n-1]]) {
	for (j=n-1;j>=0;j--) {
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

void CSVMLight::init_shrink_state(
	SHRINK_STATE *shrink_state, int32_t totdoc, int32_t maxhistory)
{
  int32_t i;

  shrink_state->deactnum=0;
  shrink_state->active = SG_MALLOC(int32_t, totdoc);
  shrink_state->inactive_since = SG_MALLOC(int32_t, totdoc);
  shrink_state->a_history = SG_MALLOC(float64_t*, maxhistory);
  shrink_state->maxhistory=maxhistory;
  shrink_state->last_lin = SG_MALLOC(float64_t, totdoc);
  shrink_state->last_a = SG_MALLOC(float64_t, totdoc);

  for (i=0;i<totdoc;i++) {
    shrink_state->active[i]=1;
    shrink_state->inactive_since[i]=0;
    shrink_state->last_a[i]=0;
    shrink_state->last_lin[i]=0;
  }
}

void CSVMLight::shrink_state_cleanup(SHRINK_STATE *shrink_state)
{
  SG_FREE(shrink_state->active);
  SG_FREE(shrink_state->inactive_since);
  if(shrink_state->deactnum > 0)
    SG_FREE((shrink_state->a_history[shrink_state->deactnum-1]));
  SG_FREE((shrink_state->a_history));
  SG_FREE((shrink_state->last_a));
  SG_FREE((shrink_state->last_lin));
}

int32_t CSVMLight::shrink_problem(
	SHRINK_STATE *shrink_state, int32_t *active2dnum,
	int32_t *last_suboptimal_at, int32_t iteration, int32_t totdoc,
	int32_t minshrink, float64_t *a, int32_t *inconsistent, float64_t* c,
	float64_t* lin, int* label)
     /* Shrink some variables away.  Do the shrinking only if at least
        minshrink variables can be removed. */
{
  int32_t i,ii,change,activenum,lastiter;
  float64_t *a_old=NULL;

  activenum=0;
  change=0;
  for (ii=0;active2dnum[ii]>=0;ii++) {
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
		  SG_INFO(" Shrinking...")
	  }

	  if (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())) { /*  non-linear case save alphas */

		  a_old=SG_MALLOC(float64_t, totdoc);
		  shrink_state->a_history[shrink_state->deactnum]=a_old;
		  for (i=0;i<totdoc;i++) {
			  a_old[i]=a[i];
		  }
	  }
	  for (ii=0;active2dnum[ii]>=0;ii++) {
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
	  if(kernel->has_property(KP_LINADD) && get_linadd_enabled())
		  shrink_state->deactnum=0;

	  if(verbosity>=2) {
		  SG_DONE()
		  SG_DEBUG("Number of inactive variables = %ld\n", totdoc-activenum)
	  }
  }
  return(activenum);
}

void* CSVMLight::reactivate_inactive_examples_linadd_helper(void* p)
{
	S_THREAD_PARAM_REACTIVATE_LINADD* params = (S_THREAD_PARAM_REACTIVATE_LINADD*) p;

	CKernel* k = params->kernel;
	float64_t* lin = params->lin;
	float64_t* last_lin = params->last_lin;
	int32_t* active = params->active;
	int32_t* docs = params->docs;
	int32_t start = params->start;
	int32_t end = params->end;

	for (int32_t i=start;i<end;i++)
	{
		if (!active[i])
			lin[i] = last_lin[i]+k->compute_optimized(docs[i]);

		last_lin[i]=lin[i];
	}

	return NULL;
}

void* CSVMLight::reactivate_inactive_examples_vanilla_helper(void* p)
{
	S_THREAD_PARAM_REACTIVATE_VANILLA* params = (S_THREAD_PARAM_REACTIVATE_VANILLA*) p;
	ASSERT(params)
	ASSERT(params->kernel)
	ASSERT(params->lin)
	ASSERT(params->aicache)
	ASSERT(params->a)
	ASSERT(params->a_old)
	ASSERT(params->changed2dnum)
	ASSERT(params->inactive2dnum)
	ASSERT(params->label)

	CKernel* k = params->kernel;
	float64_t* lin = params->lin;
	float64_t* aicache = params->aicache;
	float64_t* a= params->a;
	float64_t* a_old = params->a_old;
	int32_t* changed2dnum = params->changed2dnum;
	int32_t* inactive2dnum = params->inactive2dnum;
	int32_t* label = params->label;
	int32_t start = params->start;
	int32_t end = params->end;

	for (int32_t ii=start;ii<end;ii++)
	{
		int32_t i=changed2dnum[ii];
		int32_t j=0;
		ASSERT(i>=0)

		k->get_kernel_row(i,inactive2dnum,aicache);
		for (int32_t jj=0;(j=inactive2dnum[jj])>=0;jj++)
			lin[j]+=(a[i]-a_old[i])*aicache[j]*(float64_t)label[i];
	}
	return NULL;
}

void CSVMLight::reactivate_inactive_examples(
	int32_t* label, float64_t *a, SHRINK_STATE *shrink_state, float64_t *lin,
	float64_t *c, int32_t totdoc, int32_t iteration, int32_t *inconsistent,
	int32_t* docs, float64_t *aicache, float64_t *maxdiff)
     /* Make all variables active again which had been removed by
        shrinking. */
     /* Computes lin for those variables from scratch. */
{
  register int32_t i,j,ii,jj,t,*changed2dnum,*inactive2dnum;
  int32_t *changed,*inactive;
  register float64_t *a_old,dist;
  float64_t ex_c,target;

  if (kernel->has_property(KP_LINADD) && get_linadd_enabled()) { /* special linear case */
	  a_old=shrink_state->last_a;

	  if (!use_batch_computation || !kernel->has_property(KP_BATCHEVALUATION))
	  {
		  SG_DEBUG(" clear normal - linadd\n")
		  kernel->clear_normal();

		  int32_t num_modified=0;
		  for (i=0;i<totdoc;i++) {
			  if(a[i] != a_old[i]) {
				  kernel->add_to_normal(docs[i], ((a[i]-a_old[i])*(float64_t)label[i]));
				  a_old[i]=a[i];
				  num_modified++;
			  }
		  }

		  if (num_modified>0)
		  {
			  int32_t num_threads=parallel->get_num_threads();
			  ASSERT(num_threads>0)
			  if (num_threads < 2)
			  {
				  S_THREAD_PARAM_REACTIVATE_LINADD params;
				  params.kernel=kernel;
				  params.lin=lin;
				  params.last_lin=shrink_state->last_lin;
				  params.docs=docs;
				  params.active=shrink_state->active;
				  params.start=0;
				  params.end=totdoc;
				  reactivate_inactive_examples_linadd_helper((void*) &params);
			  }
#ifdef HAVE_PTHREAD
			  else
			  {
				  pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
				  S_THREAD_PARAM_REACTIVATE_LINADD* params = SG_MALLOC(S_THREAD_PARAM_REACTIVATE_LINADD, num_threads);
				  int32_t step= totdoc/num_threads;

				  for (t=0; t<num_threads-1; t++)
				  {
					  params[t].kernel=kernel;
					  params[t].lin=lin;
					  params[t].last_lin=shrink_state->last_lin;
					  params[t].docs=docs;
					  params[t].active=shrink_state->active;
					  params[t].start = t*step;
					  params[t].end = (t+1)*step;
					  pthread_create(&threads[t], NULL, CSVMLight::reactivate_inactive_examples_linadd_helper, (void*)&params[t]);
				  }

				  params[t].kernel=kernel;
				  params[t].lin=lin;
				  params[t].last_lin=shrink_state->last_lin;
				  params[t].docs=docs;
				  params[t].active=shrink_state->active;
				  params[t].start = t*step;
				  params[t].end = totdoc;
				  reactivate_inactive_examples_linadd_helper((void*) &params[t]);

				  for (t=0; t<num_threads-1; t++)
					  pthread_join(threads[t], NULL);

				  SG_FREE(threads);
				  SG_FREE(params);
			  }
#endif

		  }
	  }
	  else
	  {
		  float64_t *alphas = SG_MALLOC(float64_t, totdoc);
		  int32_t *idx = SG_MALLOC(int32_t, totdoc);
		  int32_t num_suppvec=0 ;

		  for (i=0; i<totdoc; i++)
		  {
			  if(a[i] != a_old[i])
			  {
				  alphas[num_suppvec] = (a[i]-a_old[i])*(float64_t)label[i];
				  idx[num_suppvec] = i ;
				  a_old[i] = a[i] ;
				  num_suppvec++ ;
			  }
		  }

		  if (num_suppvec>0)
		  {
			  int32_t num_inactive=0;
			  int32_t* inactive_idx=SG_MALLOC(int32_t, totdoc); // infact we only need a subset

			  j=0;
			  for (i=0;i<totdoc;i++)
			  {
				  if(!shrink_state->active[i])
				  {
					  inactive_idx[j++] = i;
					  num_inactive++;
				  }
			  }

			  if (num_inactive>0)
			  {
				  float64_t* dest = SG_MALLOC(float64_t, num_inactive);
				  memset(dest, 0, sizeof(float64_t)*num_inactive);

				  kernel->compute_batch(num_inactive, inactive_idx, dest, num_suppvec, idx, alphas);

				  j=0;
				  for (i=0;i<totdoc;i++) {
					  if(!shrink_state->active[i]) {
						  lin[i] = shrink_state->last_lin[i] + dest[j++] ;
					  }
					  shrink_state->last_lin[i]=lin[i];
				  }

				  SG_FREE(dest);
			  }
			  else
			  {
				  for (i=0;i<totdoc;i++)
					  shrink_state->last_lin[i]=lin[i];
			  }
			  SG_FREE(inactive_idx);
		  }
		  SG_FREE(alphas);
		  SG_FREE(idx);
	  }

	  kernel->delete_optimization();
  }
  else
  {
	  changed=SG_MALLOC(int32_t, totdoc);
	  changed2dnum=SG_MALLOC(int32_t, totdoc+11);
	  inactive=SG_MALLOC(int32_t, totdoc);
	  inactive2dnum=SG_MALLOC(int32_t, totdoc+11);
	  for (t=shrink_state->deactnum-1;(t>=0) && shrink_state->a_history[t];t--)
	  {
		  if(verbosity>=2) {
			  SG_INFO("%ld..",t)
		  }
		  a_old=shrink_state->a_history[t];
		  for (i=0;i<totdoc;i++) {
			  inactive[i]=((!shrink_state->active[i])
						   && (shrink_state->inactive_since[i] == t));
			  changed[i]= (a[i] != a_old[i]);
		  }
		  compute_index(inactive,totdoc,inactive2dnum);
		  compute_index(changed,totdoc,changed2dnum);


		  //TODO: PUT THIS BACK IN!!!!!!!! int32_t num_threads=parallel->get_num_threads();
		  int32_t num_threads=1;
		  ASSERT(num_threads>0)

		  if (num_threads < 2)
		  {
			  for (ii=0;(i=changed2dnum[ii])>=0;ii++) {
				  kernel->get_kernel_row(i,inactive2dnum,aicache);
				  for (jj=0;(j=inactive2dnum[jj])>=0;jj++)
					  lin[j]+=(a[i]-a_old[i])*aicache[j]*(float64_t)label[i];
			  }
		  }
#ifdef HAVE_PTHREAD
		  else
		  {
			  //find number of the changed ones
			  int32_t num_changed=0;
			  for (ii=0;changed2dnum[ii]>=0;ii++)
				  num_changed++;

			  if (num_changed>0)
			  {
				  pthread_t* threads= SG_MALLOC(pthread_t, num_threads-1);
				  S_THREAD_PARAM_REACTIVATE_VANILLA* params = SG_MALLOC(S_THREAD_PARAM_REACTIVATE_VANILLA, num_threads);
				  int32_t step= num_changed/num_threads;

				  // alloc num_threads many tmp buffers
				  float64_t* tmp_lin=SG_MALLOC(float64_t, totdoc*num_threads);
				  memset(tmp_lin, 0, sizeof(float64_t)*((size_t) totdoc)*num_threads);
				  float64_t* tmp_aicache=SG_MALLOC(float64_t, totdoc*num_threads);
				  memset(tmp_aicache, 0, sizeof(float64_t)*((size_t) totdoc)*num_threads);

				  int32_t thr;
				  for (thr=0; thr<num_threads-1; thr++)
				  {
					  params[thr].kernel=kernel;
					  params[thr].lin=&tmp_lin[thr*totdoc];
					  params[thr].aicache=&tmp_aicache[thr*totdoc];
					  params[thr].a=a;
					  params[thr].a_old=a_old;
					  params[thr].changed2dnum=changed2dnum;
					  params[thr].inactive2dnum=inactive2dnum;
					  params[thr].label=label;
					  params[thr].start = thr*step;
					  params[thr].end = (thr+1)*step;
					  pthread_create(&threads[thr], NULL, CSVMLight::reactivate_inactive_examples_vanilla_helper, (void*)&params[thr]);
				  }

				  params[thr].kernel=kernel;
				  params[thr].lin=&tmp_lin[thr*totdoc];
				  params[thr].aicache=&tmp_aicache[thr*totdoc];
				  params[thr].a=a;
				  params[thr].a_old=a_old;
				  params[thr].changed2dnum=changed2dnum;
				  params[thr].inactive2dnum=inactive2dnum;
				  params[thr].label=label;
				  params[thr].start = thr*step;
				  params[thr].end = num_changed;
				  reactivate_inactive_examples_vanilla_helper((void*) &params[thr]);

				  for (jj=0;(j=inactive2dnum[jj])>=0;jj++)
					  lin[j]+=tmp_lin[totdoc*thr+j];

				  for (thr=0; thr<num_threads-1; thr++)
				  {
					  pthread_join(threads[thr], NULL);

					  //add up results
					  for (jj=0;(j=inactive2dnum[jj])>=0;jj++)
						  lin[j]+=tmp_lin[totdoc*thr+j];
				  }

				  SG_FREE(tmp_lin);
				  SG_FREE(tmp_aicache);
				  SG_FREE(threads);
				  SG_FREE(params);
			  }
		  }
#endif
	  }
	  SG_FREE(changed);
	  SG_FREE(changed2dnum);
	  SG_FREE(inactive);
	  SG_FREE(inactive2dnum);
  }

  (*maxdiff)=0;
  for (i=0;i<totdoc;i++) {
    shrink_state->inactive_since[i]=shrink_state->deactnum-1;
    if(!inconsistent[i]) {
      dist=(lin[i]-model->b)*(float64_t)label[i];
      target=-(learn_parm->eps[i]-(float64_t)label[i]*c[i]);
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

  if (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())) { /* update history for non-linear */
	  for (i=0;i<totdoc;i++) {
		  (shrink_state->a_history[shrink_state->deactnum-1])[i]=a[i];
	  }
	  for (t=shrink_state->deactnum-2;(t>=0) && shrink_state->a_history[t];t--) {
		  SG_FREE(shrink_state->a_history[t]);
		  shrink_state->a_history[t]=0;
	  }
  }
}



/* start the optimizer and return the optimal values */
float64_t* CSVMLight::optimize_qp(
		QP *qp, float64_t *epsilon_crit, int32_t nx, float64_t *threshold,
		int32_t& svm_maxqpsize)
{
	register int32_t i, j, result;
	float64_t margin, obj_before, obj_after;
	float64_t sigdig, dist, epsilon_loqo;
	int32_t iter;

	if(!primal) { /* allocate memory at first call */
		primal=SG_MALLOC(float64_t, nx*3);
		dual=SG_MALLOC(float64_t, nx*2+1);
	}

	obj_before=0; /* calculate objective before optimization */
	for(i=0;i<qp->opt_n;i++) {
		obj_before+=(qp->opt_g0[i]*qp->opt_xinit[i]);
		obj_before+=(0.5*qp->opt_xinit[i]*qp->opt_xinit[i]*qp->opt_g[i*qp->opt_n+i]);
		for(j=0;j<i;j++) {
			obj_before+=(qp->opt_xinit[j]*qp->opt_xinit[i]*qp->opt_g[j*qp->opt_n+i]);
		}
	}

	result=STILL_RUNNING;
	qp->opt_ce0[0]*=(-1.0);
	/* Run pr_loqo. If a run fails, try again with parameters which lead */
	/* to a slower, but more robust setting. */
	for(margin=init_margin,iter=init_iter;
		(margin<=0.9999999) && (result!=OPTIMAL_SOLUTION);) {

		opt_precision=CMath::max(opt_precision, DEF_PRECISION);
		sigdig=-log10(opt_precision);

		result=pr_loqo((int32_t)qp->opt_n,(int32_t)qp->opt_m,
				(float64_t *)qp->opt_g0,(float64_t *)qp->opt_g,
				(float64_t *)qp->opt_ce,(float64_t *)qp->opt_ce0,
				(float64_t *)qp->opt_low,(float64_t *)qp->opt_up,
				(float64_t *)primal,(float64_t *)dual,
				(int32_t)(verbosity-2),
				(float64_t)sigdig,(int32_t)iter,
				(float64_t)margin,(float64_t)(qp->opt_up[0])/4.0,(int32_t)0);

		if(CMath::is_nan(dual[0])) {     /* check for choldc problem */
			if(verbosity>=2) {
				SG_SDEBUG("Restarting PR_LOQO with more conservative parameters.\n")
			}
			if(init_margin<0.80) { /* become more conservative in general */
				init_margin=(4.0*margin+1.0)/5.0;
			}
			margin=(margin+1.0)/2.0;
			(opt_precision)*=10.0;   /* reduce precision */
			if(verbosity>=2) {
				SG_SDEBUG("Reducing precision of PR_LOQO.\n")
			}
		}
		else if(result!=OPTIMAL_SOLUTION) {
			iter+=2000;
			init_iter+=10;
			(opt_precision)*=10.0;   /* reduce precision */
			if(verbosity>=2) {
				SG_SDEBUG("Reducing precision of PR_LOQO due to (%ld).\n",result)
			}
		}
	}

	if(qp->opt_m)         /* Thanks to Alex Smola for this hint */
		model_b=dual[0];
	else
		model_b=0;

	/* Check the precision of the alphas. If results of current optimization */
	/* violate KT-Conditions, relax the epsilon on the bounds on alphas. */
	epsilon_loqo=1E-10;
	for(i=0;i<qp->opt_n;i++) {
		dist=-model_b*qp->opt_ce[i];
		dist+=(qp->opt_g0[i]+1.0);
		for(j=0;j<i;j++) {
			dist+=(primal[j]*qp->opt_g[j*qp->opt_n+i]);
		}
		for(j=i;j<qp->opt_n;j++) {
			dist+=(primal[j]*qp->opt_g[i*qp->opt_n+j]);
		}
		/*  SG_SDEBUG("LOQO: a[%d]=%f, dist=%f, b=%f\n",i,primal[i],dist,dual[0]) */
		if((primal[i]<(qp->opt_up[i]-epsilon_loqo)) && (dist < (1.0-(*epsilon_crit)))) {
			epsilon_loqo=(qp->opt_up[i]-primal[i])*2.0;
		}
		else if((primal[i]>(0+epsilon_loqo)) && (dist > (1.0+(*epsilon_crit)))) {
			epsilon_loqo=primal[i]*2.0;
		}
	}

	for(i=0;i<qp->opt_n;i++) {  /* clip alphas to bounds */
		if(primal[i]<=(0+epsilon_loqo)) {
			primal[i]=0;
		}
		else if(primal[i]>=(qp->opt_up[i]-epsilon_loqo)) {
			primal[i]=qp->opt_up[i];
		}
	}

	obj_after=0;  /* calculate objective after optimization */
	for(i=0;i<qp->opt_n;i++) {
		obj_after+=(qp->opt_g0[i]*primal[i]);
		obj_after+=(0.5*primal[i]*primal[i]*qp->opt_g[i*qp->opt_n+i]);
		for(j=0;j<i;j++) {
			obj_after+=(primal[j]*primal[i]*qp->opt_g[j*qp->opt_n+i]);
		}
	}

	/* if optimizer returned NAN values, reset and retry with smaller */
	/* working set. */
	if(CMath::is_nan(obj_after) || CMath::is_nan(model_b)) {
		for(i=0;i<qp->opt_n;i++) {
			primal[i]=qp->opt_xinit[i];
		}
		model_b=0;
		if(svm_maxqpsize>2) {
			svm_maxqpsize--;  /* decrease size of qp-subproblems */
		}
	}

	if(obj_after >= obj_before) { /* check whether there was progress */
		(opt_precision)/=100.0;
		precision_violations++;
		if(verbosity>=2) {
			SG_SDEBUG("Increasing Precision of PR_LOQO.\n")
		}
	}

	// TODO: add parameter for this (e.g. for MTL experiments)
	if(precision_violations > 5000) {
		(*epsilon_crit)*=10.0;
		precision_violations=0;
		SG_SINFO("Relaxing epsilon on KT-Conditions.\n")
	}

	(*threshold)=model_b;

	if(result!=OPTIMAL_SOLUTION) {
		SG_SERROR("PR_LOQO did not converge.\n")
		return(qp->opt_xinit);
	}
	else {
		return(primal);
	}
}


#endif //USE_SVMLIGHT

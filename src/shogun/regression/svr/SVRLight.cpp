/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/config.h>

#ifdef USE_SVMLIGHT

#include <io/SGIO.h>
#include <mathematics/lapack.h>
#include <lib/Signal.h>
#include <mathematics/Math.h>
#include <regression/svr/SVRLight.h>
#include <machine/KernelMachine.h>
#include <kernel/CombinedKernel.h>
#include <labels/RegressionLabels.h>

#include <unistd.h>

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}
#endif

#include <base/Parallel.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct S_THREAD_PARAM_SVRLIGHT
{
	float64_t* lin;
	int32_t start, end;
	int32_t* active2dnum;
	int32_t* docs;
	CKernel* kernel;
    int32_t num_vectors;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

CSVRLight::CSVRLight(float64_t C, float64_t eps, CKernel* k, CLabels* lab)
: CSVMLight(C, k, lab)
{
	set_tube_epsilon(eps);
}

CSVRLight::CSVRLight()
: CSVMLight()
{
}

/** default destructor */
CSVRLight::~CSVRLight()
{
}

EMachineType CSVRLight::get_classifier_type()
{
	return CT_SVRLIGHT;
}

bool CSVRLight::train_machine(CFeatures* data)
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
	learn_parm->transduction_posratio=0.33;
	learn_parm->svm_costratio=get_C2()/get_C1();
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=epsilon; // GU: better decrease it ... ??
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;

	if (!kernel)
	{
		SG_ERROR("SVR_light can not proceed without kernel!\n")
		return false ;
	}

	if (!m_labels)
	{
		SG_ERROR("SVR_light can not proceed without labels!\n")
		return false;
	}

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")
		kernel->init(data, data);
	}

	if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
		kernel->clear_normal();

	// output some info
	SG_DEBUG("qpsize = %i\n", learn_parm->svm_maxqpsize)
	SG_DEBUG("epsilon = %1.1e\n", learn_parm->epsilon_crit)
	SG_DEBUG("kernel->has_property(KP_LINADD) = %i\n", kernel->has_property(KP_LINADD))
	SG_DEBUG("kernel->has_property(KP_KERNCOMBINATION) = %i\n", kernel->has_property(KP_KERNCOMBINATION))
	SG_DEBUG("get_linadd_enabled() = %i\n", get_linadd_enabled())
	SG_DEBUG("kernel->get_num_subkernels() = %i\n", kernel->get_num_subkernels())

	use_kernel_cache = !((kernel->get_kernel_type() == K_CUSTOM) ||
						 (get_linadd_enabled() && kernel->has_property(KP_LINADD)));

	SG_DEBUG("use_kernel_cache = %i\n", use_kernel_cache)

	// train the svm
	svr_learn();

	// brain damaged svm light work around
	create_new_model(model->sv_num-1);
	set_bias(-model->b);
	for (int32_t i=0; i<model->sv_num-1; i++)
	{
		set_alpha(i, model->alpha[i+1]);
		set_support_vector(i, model->supvec[i+1]);
	}

	if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
		kernel->clear_normal() ;

	return true ;
}

void CSVRLight::svr_learn()
{
	int32_t *inconsistent, i, j;
	int32_t upsupvecnum;
	float64_t maxdiff, *lin, *c, *a;
	int32_t iterations;
	float64_t *xi_fullset; /* buffer for storing xi on full sample in loo */
	float64_t *a_fullset;  /* buffer for storing alpha on full sample in loo */
	TIMING timing_profile;
	SHRINK_STATE shrink_state;
	int32_t* label;
	int32_t* docs;

	ASSERT(m_labels)
	int32_t totdoc=m_labels->get_num_labels();
	num_vectors=totdoc;

	// set up regression problem in standard form
	docs=SG_MALLOC(int32_t, 2*totdoc);
	label=SG_MALLOC(int32_t, 2*totdoc);
	c = SG_MALLOC(float64_t, 2*totdoc);

  for(i=0;i<totdoc;i++) {
	  docs[i]=i;
	  j=2*totdoc-1-i;
	  label[i]=+1;
	  c[i]=((CRegressionLabels*) m_labels)->get_label(i);
	  docs[j]=j;
	  label[j]=-1;
	  c[j]=((CRegressionLabels*) m_labels)->get_label(i);
  }
  totdoc*=2;

  //prepare kernel cache for regression (i.e. cachelines are twice of current size)
  kernel->resize_kernel_cache( kernel->get_cache_size(), true);

  if (kernel->get_kernel_type() == K_COMBINED)
  {
	  CCombinedKernel* k = (CCombinedKernel*) kernel;

	  for (index_t k_idx=0; k_idx<k->get_num_kernels(); k_idx++)
	  {
		  CKernel* kn = k->get_kernel(k_idx);
		  kn->resize_kernel_cache( kernel->get_cache_size(), true);
		  SG_UNREF(kn);
	  }
  }

  timing_profile.time_kernel=0;
  timing_profile.time_opti=0;
  timing_profile.time_shrink=0;
  timing_profile.time_update=0;
  timing_profile.time_model=0;
  timing_profile.time_check=0;
  timing_profile.time_select=0;

	SG_FREE(W);
	W=NULL;

	if (kernel->has_property(KP_KERNCOMBINATION) && callback)
	{
		W = SG_MALLOC(float64_t, totdoc*kernel->get_num_subkernels());
		for (i=0; i<totdoc*kernel->get_num_subkernels(); i++)
			W[i]=0;
	}

	/* make sure -n value is reasonable */
	if((learn_parm->svm_newvarsinqp < 2)
			|| (learn_parm->svm_newvarsinqp > learn_parm->svm_maxqpsize)) {
		learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;
	}

	init_shrink_state(&shrink_state,totdoc,(int32_t)MAXSHRINK);

	inconsistent = SG_MALLOC(int32_t, totdoc);
	a = SG_MALLOC(float64_t, totdoc);
	a_fullset = SG_MALLOC(float64_t, totdoc);
	xi_fullset = SG_MALLOC(float64_t, totdoc);
	lin = SG_MALLOC(float64_t, totdoc);
	learn_parm->svm_cost = SG_MALLOC(float64_t, totdoc);
	if (m_linear_term.vlen>0)
		learn_parm->eps=get_linear_term_array();
	else
	{
		learn_parm->eps=SG_MALLOC(float64_t, totdoc);      /* equivalent regression epsilon for classification */
		SGVector<float64_t>::fill_vector(learn_parm->eps, totdoc, tube_epsilon);
	}

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

  for(i=0;i<totdoc;i++) {    /* various inits */
    inconsistent[i]=0;
    a[i]=0;
    lin[i]=0;

		if(label[i] > 0) {
			learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
				fabs((float64_t)label[i]);
		}
		else if(label[i] < 0) {
			learn_parm->svm_cost[i]=learn_parm->svm_c*fabs((float64_t)label[i]);
		}
		else
			ASSERT(false)
	}

	if(verbosity==1) {
		SG_DEBUG("Optimizing...\n")
	}

	/* train the svm */
		SG_DEBUG("num_train: %d\n", totdoc)
  iterations=optimize_to_convergence(docs,label,totdoc,
                     &shrink_state,inconsistent,a,lin,
                     c,&timing_profile,
                     &maxdiff,(int32_t)-1,
                     (int32_t)1);


	if(verbosity>=1) {
		SG_DONE()
		SG_INFO("(%ld iterations)\n",iterations)
		SG_INFO("Optimization finished (maxdiff=%.8f).\n",maxdiff)
		SG_INFO("obj = %.16f, rho = %.16f\n",get_objective(),model->b)

		upsupvecnum=0;

		SG_DEBUG("num sv: %d\n", model->sv_num)
		for(i=1;i<model->sv_num;i++)
		{
			if(fabs(model->alpha[i]) >=
					(learn_parm->svm_cost[model->supvec[i]]-
					 learn_parm->epsilon_a))
				upsupvecnum++;
		}
		SG_INFO("Number of SV: %d (including %d at upper bound)\n",
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
	SG_FREE(label);
	SG_FREE(inconsistent);
	SG_FREE(c);
	SG_FREE(a);
	SG_FREE(a_fullset);
	SG_FREE(xi_fullset);
	SG_FREE(lin);
	SG_FREE(learn_parm->svm_cost);
	SG_FREE(docs);
}

float64_t CSVRLight::compute_objective_function(
	float64_t *a, float64_t *lin, float64_t *c, float64_t* eps, int32_t *label,
	int32_t totdoc)
{
  /* calculate value of objective function */
  float64_t criterion=0;

  for(int32_t i=0;i<totdoc;i++)
	  criterion+=(eps[i]-(float64_t)label[i]*c[i])*a[i]+0.5*a[i]*label[i]*lin[i];

  /* float64_t check=0;
  for(int32_t i=0;i<totdoc;i++)
  {
	  check+=a[i]*eps-a[i]*label[i]*c[i];
	  for(int32_t j=0;j<totdoc;j++)
		  check+= 0.5*a[i]*label[i]*a[j]*label[j]*compute_kernel(i,j);

  }

  SG_INFO("REGRESSION OBJECTIVE %f vs. CHECK %f (diff %f)\n", criterion, check, criterion-check) */

  return(criterion);
}

void* CSVRLight::update_linear_component_linadd_helper(void *params_)
{
	S_THREAD_PARAM_SVRLIGHT * params = (S_THREAD_PARAM_SVRLIGHT*) params_ ;

	int32_t jj=0, j=0 ;

	for(jj=params->start;(jj<params->end) && (j=params->active2dnum[jj])>=0;jj++)
		params->lin[j]+=params->kernel->compute_optimized(CSVRLight::regression_fix_index2(params->docs[j], params->num_vectors));

	return NULL ;
}

int32_t CSVRLight::regression_fix_index(int32_t i)
{
	if (i>=num_vectors)
		i=2*num_vectors-1-i;

	return i;
}

int32_t CSVRLight::regression_fix_index2(
		int32_t i, int32_t num_vectors)
{
	if (i>=num_vectors)
		i=2*num_vectors-1-i;

	return i;
}

float64_t CSVRLight::compute_kernel(int32_t i, int32_t j)
{
	i=regression_fix_index(i);
	j=regression_fix_index(j);
	return kernel->kernel(i, j);
}

void CSVRLight::update_linear_component(
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
			update_linear_component_mkl_linadd(docs, label, active2dnum, a, a_old, working2dnum,
											   totdoc,	lin, aicache, c) ;
		}
		else
		{
			kernel->clear_normal();

			int32_t num_working=0;
			for(ii=0;(i=working2dnum[ii])>=0;ii++) {
				if(a[i] != a_old[i]) {
					kernel->add_to_normal(regression_fix_index(docs[i]), (a[i]-a_old[i])*(float64_t)label[i]);
					num_working++;
				}
			}

			if (num_working>0)
			{
				if (parallel->get_num_threads() < 2)
				{
					for(jj=0;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=kernel->compute_optimized(regression_fix_index(docs[j]));
					}
				}
#ifdef HAVE_PTHREAD
				else
				{
					int32_t num_elem = 0 ;
					for(jj=0;(j=active2dnum[jj])>=0;jj++) num_elem++ ;

					pthread_t* threads = SG_MALLOC(pthread_t, parallel->get_num_threads()-1);
					S_THREAD_PARAM_SVRLIGHT* params = SG_MALLOC(S_THREAD_PARAM_SVRLIGHT, parallel->get_num_threads()-1);
					int32_t start = 0 ;
					int32_t step = num_elem/parallel->get_num_threads() ;
					int32_t end = step ;

					for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
					{
						params[t].kernel = kernel ;
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

					for(jj=params[parallel->get_num_threads()-2].end;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=kernel->compute_optimized(regression_fix_index(docs[j]));
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
					a, a_old, working2dnum, totdoc,	lin, aicache, c) ;
		}
		else {
			for(jj=0;(i=working2dnum[jj])>=0;jj++) {
				if(a[i] != a_old[i]) {
					kernel->get_kernel_row(i,active2dnum,aicache);
					for(ii=0;(j=active2dnum[ii])>=0;ii++)
						lin[j]+=(a[i]-a_old[i])*aicache[j]*(float64_t)label[i];
				}
			}
		}
	}
}

void CSVRLight::update_linear_component_mkl(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache, float64_t* c)
{
	int32_t num         = totdoc;
	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	const float64_t* old_beta  = kernel->get_subkernel_weights(num_weights);

	ASSERT(num_weights==num_kernels)

	if ((kernel->get_kernel_type()==K_COMBINED) &&
			 (!((CCombinedKernel*)kernel)->get_append_subkernel_weights()))// for combined kernel
	{
		CCombinedKernel* k = (CCombinedKernel*) kernel;

		int32_t n = 0, i, j ;

		for (index_t k_idx=0; k_idx<k->get_num_kernels(); k_idx++)
		{
			CKernel* kn = k->get_kernel(k_idx);
			for(i=0;i<num;i++)
			{
				if(a[i] != a_old[i])
				{
					kn->get_kernel_row(i,NULL,aicache, true);
					for(j=0;j<num;j++)
						W[j*num_kernels+n]+=(a[i]-a_old[i])*aicache[regression_fix_index(j)]*(float64_t)label[i];
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
			kernel->set_subkernel_weights(SGVector<float64_t>(w1, num_weights)) ;

			for(int32_t i=0;i<num;i++)
			{
				if(a[i] != a_old[i])
				{
					for(int32_t j=0;j<num;j++)
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

	call_mkl_callback(a, label, lin, c, totdoc);
}


void CSVRLight::update_linear_component_mkl_linadd(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache, float64_t* c)
{
	// kernel with LP_LINADD property is assumed to have
	// compute_by_subkernel functions
	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	const float64_t* old_beta   = kernel->get_subkernel_weights(num_weights);

	ASSERT(num_weights==num_kernels)

	float64_t* w_backup=SG_MALLOC(float64_t, num_kernels);
	float64_t* w1=SG_MALLOC(float64_t, num_kernels);

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
	for(int32_t ii=0, i=0;(i=working2dnum[ii])>=0;ii++) {
		if(a[i] != a_old[i]) {
			kernel->add_to_normal(regression_fix_index(docs[i]), (a[i]-a_old[i])*(float64_t)label[i]);
		}
	}

	// determine contributions of different kernels
	for (int32_t i=0; i<num_vectors; i++)
		kernel->compute_by_subkernel(i,&W[i*num_kernels]) ;

	// restore old weights
	kernel->set_subkernel_weights(SGVector<float64_t>(w_backup,num_weights));

	call_mkl_callback(a, label, lin, c, totdoc);
}

void CSVRLight::call_mkl_callback(float64_t* a, int32_t* label, float64_t* lin, float64_t* c, int32_t totdoc)
{
	int32_t num = totdoc;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	float64_t sumalpha = 0;
	float64_t* sumw=SG_MALLOC(float64_t, num_kernels);

	for (int32_t i=0; i<num; i++)
		sumalpha-=a[i]*(learn_parm->eps[i]-label[i]*c[i]);

#ifdef HAVE_LAPACK
	int nk = (int) num_kernels; // calling external lib
	double* alphay  = SG_MALLOC(double, num);
	for (int32_t i=0; i<num; i++)
		alphay[i]=a[i]*label[i];

	for (int32_t i=0; i<num_kernels; i++)
		sumw[i]=0;

	cblas_dgemv(CblasColMajor, CblasNoTrans, nk, (int) num, 0.5, (double*) W,
		nk, (double*) alphay, 1, 1.0, (double*) sumw, 1);

	SG_FREE(alphay);
#else
	for (int32_t d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for(int32_t i=0; i<num; i++)
			sumw[d] += 0.5*a[i]*label[i]*W[i*num_kernels+d];
	}
#endif

	if (callback)
		mkl_converged=callback(mkl, sumw, sumalpha);

	const float64_t* new_beta   = kernel->get_subkernel_weights(num_kernels);

	// update lin
#ifdef HAVE_LAPACK
	cblas_dgemv(CblasColMajor, CblasTrans, nk, (int) num, 1.0, (double*) W,
		nk, (double*) new_beta, 1, 0.0, (double*) lin, 1);
#else
	for(int32_t i=0; i<num; i++)
		lin[i]=0 ;
	for (int32_t d=0; d<num_kernels; d++)
		if (new_beta[d]!=0)
			for(int32_t i=0; i<num; i++)
				lin[i] += new_beta[d]*W[i*num_kernels+d] ;
#endif


	SG_FREE(sumw);
}


void CSVRLight::reactivate_inactive_examples(
	int32_t* label, float64_t *a, SHRINK_STATE *shrink_state, float64_t *lin,
	float64_t *c, int32_t totdoc, int32_t iteration, int32_t *inconsistent,
	int32_t* docs, float64_t *aicache, float64_t *maxdiff)
     /* Make all variables active again which had been removed by
        shrinking. */
     /* Computes lin for those variables from scratch. */
{
  register int32_t i=0,j,ii=0,jj,t,*changed2dnum,*inactive2dnum;
  int32_t *changed,*inactive;
  register float64_t *a_old,dist;
  float64_t ex_c,target;

  if (kernel->has_property(KP_LINADD) && get_linadd_enabled()) { /* special linear case */
	  a_old=shrink_state->last_a;

	  kernel->clear_normal();
	  int32_t num_modified=0;
	  for(i=0;i<totdoc;i++) {
		  if(a[i] != a_old[i]) {
			  kernel->add_to_normal(regression_fix_index(docs[i]), ((a[i]-a_old[i])*(float64_t)label[i]));
			  a_old[i]=a[i];
			  num_modified++;
		  }
	  }

	  if (num_modified>0)
	  {
		  for(i=0;i<totdoc;i++) {
			  if(!shrink_state->active[i]) {
				  lin[i]=shrink_state->last_lin[i]+kernel->compute_optimized(regression_fix_index(docs[i]));
			  }
			  shrink_state->last_lin[i]=lin[i];
		  }
	  }
  }
  else
  {
	  changed=SG_MALLOC(int32_t, totdoc);
	  changed2dnum=SG_MALLOC(int32_t, totdoc+11);
	  inactive=SG_MALLOC(int32_t, totdoc);
	  inactive2dnum=SG_MALLOC(int32_t, totdoc+11);
	  for(t=shrink_state->deactnum-1;(t>=0) && shrink_state->a_history[t];t--) {
		  if(verbosity>=2) {
			  SG_INFO("%ld..",t)
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
			  CKernelMachine::kernel->get_kernel_row(i,inactive2dnum,aicache);
			  for(jj=0;(j=inactive2dnum[jj])>=0;jj++)
				  lin[j]+=(a[i]-a_old[i])*aicache[j]*(float64_t)label[i];
		  }
	  }
	  SG_FREE(changed);
	  SG_FREE(changed2dnum);
	  SG_FREE(inactive);
	  SG_FREE(inactive2dnum);
  }

  (*maxdiff)=0;
  for(i=0;i<totdoc;i++) {
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
  if (use_kernel_cache) { /* update history for non-linear */
	  for(i=0;i<totdoc;i++) {
		  (shrink_state->a_history[shrink_state->deactnum-1])[i]=a[i];
	  }
	  for(t=shrink_state->deactnum-2;(t>=0) && shrink_state->a_history[t];t--) {
		  SG_FREE(shrink_state->a_history[t]);
		  shrink_state->a_history[t]=0;
	  }
  }
}
#endif //USE_SVMLIGHT

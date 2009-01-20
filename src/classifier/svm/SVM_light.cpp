/***********************************************************************/
/*                                                                     */
/*   SVM_light.cpp                                                     */
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
#include "lib/config.h"

#ifdef USE_SVMLIGHT

#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Mathematics.h"
#include "lib/Time.h"
#include "lib/lapack.h"

#include "features/WordFeatures.h"
#include "classifier/svm/SVM_light.h"
#include "classifier/svm/Optimizer.h"

#include "kernel/Kernel.h"
#include "kernel/KernelMachine.h"
#include "kernel/CombinedKernel.h"
#include "kernel/AUCKernel.h"

#include <unistd.h>

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}
#endif

#include "base/Parallel.h"

#ifndef WIN32
#include <pthread.h>
#endif

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

struct S_THREAD_PARAM 
{
	float64_t * lin ;
	float64_t* W;
	int32_t start, end;
	int32_t * active2dnum ;
	int32_t * docs ;
	CKernel* kernel ;
}  ;

struct S_THREAD_PARAM_KERNEL 
{
	float64_t *Kval ;
	int32_t *KI, *KJ ;
	int32_t start, end;
    CKernel * kernel;
}  ;

void* CSVMLight::update_linear_component_linadd_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;
	
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
		params->Kval[jj]=params->kernel->kernel(params->KI[jj], params->KJ[jj]) ;

	return NULL ;
}

void* CSVMLight::update_linear_component_mkl_linadd_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;

	int32_t num_kernels=params->kernel->get_num_subkernels();

	// determine contributions of different kernels
	for (int32_t i=params->start; i<params->end; i++)
		params->kernel->compute_by_subkernel(i,&(params->W[i*num_kernels]));

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
	W=NULL;
	model=new MODEL[1];
	learn_parm=new LEARN_PARM[1];
	model->supvec=NULL;
	model->alpha=NULL;
	model->index=NULL;

	//certain setup params
	verbosity=1;
	init_margin=0.15;
	init_iter=500;
	precision_violations=0;
	opt_precision=DEF_PRECISION;

	// MKL stuff
	rho=0 ;
	mymaxdiff=1 ;
	weight_epsilon=0 ;
	lp_C = 0 ;
	
#ifdef USE_CPLEX
	lp_cplex = NULL ;
	env = NULL ;
#endif

#ifdef USE_GLPK
	lp_glpk = NULL;
#endif

	lp_initialized = false ;
}

#ifdef USE_CPLEX
bool CSVMLight::init_cplex()
{
	while (env==NULL)
	{
		SG_INFO( "trying to initialize CPLEX\n") ;

		int status = 0; /* calling external lib */
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
			status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, 2);
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

bool CSVMLight::cleanup_cplex()
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
bool CSVMLight::init_glpk()
{
	lp_glpk = lpx_create_prob();
	lpx_set_obj_dir(lp_glpk, LPX_MIN);
	glp_term_out(GLP_OFF);
	return (lp_glpk != NULL);
}

bool CSVMLight::cleanup_glpk()
{
	lp_initialized = false;
	if (lp_glpk)
		lpx_delete_prob(lp_glpk);
	lp_glpk = NULL;
	return true;
}

bool CSVMLight::check_lpx_status(LPX *lp)
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

CSVMLight::~CSVMLight()
{

  delete[] model->supvec;
  delete[] model->alpha;
  delete[] model->index;
  delete[] model;
  delete[] learn_parm;

  // MKL stuff
  delete[] W ;
}

bool CSVMLight::setup_auc_maximization()
{
	SG_INFO( "setting up AUC maximization\n") ;
	
	// get the original labels
	int32_t num=0;
	ASSERT(labels);
	int32_t* int_labels=labels->get_int_labels(num);
	ASSERT(kernel->get_num_vec_rhs()==num);
	
	// count positive and negative
	int32_t num_pos = 0 ;
	int32_t num_neg = 0 ;
	for (int32_t i=0; i<num; i++)
		if (int_labels[i]==1)
			num_pos++ ;
		else 
			num_neg++ ;
	
	// create AUC features and labels (alternate labels)
	int32_t num_auc = num_pos*num_neg ;
	SG_INFO( "num_pos: %i  num_neg: %i  num_auc: %i\n", num_pos, num_neg, num_auc) ;

	uint16_t* features_auc = new uint16_t[num_auc*2] ;
	int32_t* labels_auc = new int32_t[num_auc] ;
	int32_t n=0 ;
	for (int32_t i=0; i<num; i++)
		if (int_labels[i]==1)
			for (int32_t j=0; j<num; j++)
				if (int_labels[j]==-1)
				{
					if (n%2==0)
					{
						features_auc[n*2]=i ;
						features_auc[n*2+1]=j ;
						labels_auc[n] = 1 ;
					}
					else
					{
						features_auc[n*2]=j ;
						features_auc[n*2+1]=i ;
						labels_auc[n] = -1 ;
					}
					n++ ;
					ASSERT(n<=num_auc);
				}

	// create label object and attach it to svm
	CLabels* lab_auc = new CLabels(num_auc) ;
	lab_auc->set_int_labels(labels_auc, num_auc) ;
	set_labels(lab_auc);
	
	// create feature object
	CWordFeatures* f = new CWordFeatures((int32_t)0,0) ;
	f->set_feature_matrix(features_auc, 2, num_auc) ;

	// create AUC kernel and attach the features
	CAUCKernel* k= new CAUCKernel(10, kernel) ;
	kernel->init(f,f);

	set_kernel(k) ;

	delete[] int_labels ;
	delete[] labels_auc ;

	return true ;
}

bool CSVMLight::train()
{
	//certain setup params	
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
	learn_parm->eps=-1.0;      /* equivalent regression epsilon for classification */
	learn_parm->transduction_posratio=0.33;
	learn_parm->svm_costratio=C2/C1;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=epsilon; // GU: better decrease it ... ??
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;

    if (!kernel || !kernel->has_features())
    {
        SG_ERROR( "SVM_light can not proceed without initialized kernel!\n");
        return false ;
    }
	ASSERT(labels && labels->get_num_labels());
	ASSERT(labels->is_two_class_labeling());
	ASSERT(kernel->get_num_vec_lhs()==labels->get_num_labels());

	if (weight_epsilon<=0)
		weight_epsilon=1e-2 ;

	// in case of LINADD enabled kernels cleanup!
	if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
		kernel->clear_normal() ;

	// output some info
	SG_DEBUG( "threads = %i\n", parallel.get_num_threads()) ;
	SG_DEBUG( "qpsize = %i\n", learn_parm->svm_maxqpsize) ;
	SG_DEBUG( "epsilon = %1.1e\n", learn_parm->epsilon_crit) ;
	SG_DEBUG( "weight_epsilon = %1.1e\n", weight_epsilon) ;
	SG_DEBUG( "C_mkl = %1.1e\n", C_mkl) ;
	SG_DEBUG( "mkl_norm = %1.3e\n", mkl_norm);
	SG_DEBUG( "kernel->has_property(KP_LINADD) = %i\n", kernel->has_property(KP_LINADD)) ;
	SG_DEBUG( "kernel->has_property(KP_KERNCOMBINATION) = %i\n", kernel->has_property(KP_KERNCOMBINATION)) ;
	SG_DEBUG( "kernel->has_property(KP_BATCHEVALUATION) = %i\n", kernel->has_property(KP_BATCHEVALUATION)) ;
	SG_DEBUG( "kernel->get_optimization_type() = %s\n", kernel->get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" : "SLOWBUTMEMEFFICIENT" ) ;
	SG_DEBUG( "get_solver_type() = %i\n", get_solver_type());
	SG_DEBUG( "get_mkl_enabled() = %i\n", get_mkl_enabled()) ;
	SG_DEBUG( "get_linadd_enabled() = %i\n", get_linadd_enabled()) ;
	SG_DEBUG( "get_batch_computation_enabled() = %i\n", get_batch_computation_enabled()) ;
	SG_DEBUG( "kernel->get_num_subkernels() = %i\n", kernel->get_num_subkernels()) ;
	SG_DEBUG( "estimated time: %1.1f minutes\n", 5e-11*pow(kernel->get_num_subkernels(),2.22)*pow(kernel->get_num_vec_rhs(),1.68)*pow(CMath::log2(1/weight_epsilon),2.52)/60) ;

	use_kernel_cache = !((kernel->get_kernel_type() == K_CUSTOM) ||
						 (get_linadd_enabled() && kernel->has_property(KP_LINADD)));

	SG_DEBUG( "use_kernel_cache = %i\n", use_kernel_cache) ;

#ifdef USE_CPLEX
	cleanup_cplex();

	if (get_mkl_enabled() &&
			(get_solver_type()==ST_CPLEX || get_solver_type()==ST_AUTO))
		init_cplex();
#endif

#ifdef USE_GLPK
	cleanup_glpk();

	if (get_mkl_enabled() && ( get_solver_type() == ST_GLPK ||
				( mkl_norm == 1 && get_solver_type()==ST_AUTO)))
		init_glpk();
#endif
	
	if (kernel->get_kernel_type() == K_COMBINED)
	{
		CKernel* kn = ((CCombinedKernel*)kernel)->get_first_kernel();

		while (kn)
		{
			// allocate kernel cache but clean up beforehand
			kn->resize_kernel_cache(kn->get_cache_size());
			kn = ((CCombinedKernel*) kernel)->get_next_kernel(kn) ;
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

#ifdef USE_CPLEX
	cleanup_cplex();
#endif
	
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
	int32_t inconsistentnum;
	int32_t misclassified,upsupvecnum;
	float64_t maxdiff, *lin, *c, *a;
	int32_t runtime_start,runtime_end;
	int32_t iterations;
	int32_t trainpos=0, trainneg=0 ;
	int32_t totdoc=0;
	ASSERT(labels);
	int32_t* label=labels->get_int_labels(totdoc);
	ASSERT(label);
	int32_t* docs=new int32_t[totdoc];
	delete[] W;
	W=NULL;
	rho=0 ;
	w_gap = 1 ;
	count = 0 ;

	if (kernel->has_property(KP_KERNCOMBINATION))
	{
		W = new float64_t[totdoc*kernel->get_num_subkernels()];
		for (i=0; i<totdoc*kernel->get_num_subkernels(); i++)
			W[i]=0;
	}

	for (i=0; i<totdoc; i++)
		docs[i]=i;

	float64_t *xi_fullset; /* buffer for storing xi on full sample in loo */
	float64_t *a_fullset;  /* buffer for storing alpha on full sample in loo */
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

	init_shrink_state(&shrink_state,totdoc,(int32_t)MAXSHRINK);

	inconsistent = new int32_t[totdoc];
	c = new float64_t[totdoc];
	a = new float64_t[totdoc];
	a_fullset = new float64_t[totdoc];
	xi_fullset = new float64_t[totdoc];
	lin = new float64_t[totdoc];
	learn_parm->svm_cost = new float64_t[totdoc];

	delete[] model->supvec;
	delete[] model->alpha;
	delete[] model->index;
	model->supvec = new int32_t[totdoc+2];
	model->alpha = new float64_t[totdoc+2];
	model->index = new int32_t[totdoc+2];

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
	inconsistentnum=0;


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
	SG_DEBUG( "alpha:%d num_sv:%d\n", svm_model.alpha, get_num_support_vectors());
  if(svm_model.alpha && get_num_support_vectors()) {
    if(verbosity>=1) {
		SG_INFO( "Computing starting state...");
    }

	float64_t* alpha = new float64_t[totdoc];

	for (i=0; i<totdoc; i++)
		alpha[i]=0;

	for (i=0; i<get_num_support_vectors(); i++)
		alpha[get_support_vector(i)]=get_alpha(i);
	
    int32_t* index = new int32_t[totdoc];
    int32_t* index2dnum = new int32_t[totdoc+11];
    float64_t* aicache = new float64_t[totdoc];
    for (i=0;i<totdoc;i++) {    /* create full index and clip alphas */
      index[i]=1;
      alpha[i]=fabs(alpha[i]);
      if(alpha[i]<0) alpha[i]=0;
      if(alpha[i]>learn_parm->svm_cost[i]) alpha[i]=learn_parm->svm_cost[i];
    }

	if (use_kernel_cache)
	{
		if ( kernel->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() &&
				(!((CCombinedKernel*)kernel)->get_append_subkernel_weights()) 
		   )
		{
			CCombinedKernel* k      = (CCombinedKernel*) kernel;
			CKernel* kn = k->get_first_kernel();

			while (kn)
			{
				for (i=0;i<totdoc;i++)     /* fill kernel cache with unbounded SV */
					if((alpha[i]>0) && (alpha[i]<learn_parm->svm_cost[i]) 
							&& (kn->kernel_cache_space_available())) 
						kn->cache_kernel_row(i);

				for (i=0;i<totdoc;i++)     /* fill rest of kernel cache with bounded SV */
					if((alpha[i]==learn_parm->svm_cost[i]) 
							&& (kn->kernel_cache_space_available())) 
						kn->cache_kernel_row(i);

				kn = k->get_next_kernel(kn) ;
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
    (void)compute_index(index,totdoc,index2dnum);
    update_linear_component(docs,label,index2dnum,alpha,a,index2dnum,totdoc,
			    lin,aicache,NULL);
    (void)calculate_svm_model(docs,label,lin,alpha,a,c,
			      index2dnum,index2dnum);
    for (i=0;i<totdoc;i++) {    /* copy initial alphas */
      a[i]=alpha[i];
    }

    delete[] index;
    delete[] index2dnum;
    delete[] aicache;
    delete[] alpha;

    if(verbosity>=1)
		SG_DONE();
  } 
		SG_DEBUG( "%d totdoc %d pos %d neg\n", totdoc, trainpos, trainneg);
		SG_DEBUG( "Optimizing...\n");

	/* train the svm */
  iterations=optimize_to_convergence(docs,label,totdoc,
                     &shrink_state,inconsistent,a,lin,
                     c,&timing_profile,
                     &maxdiff,(int32_t)-1,
                     (int32_t)1);


	if(verbosity>=1) {
		if(verbosity==1)
		{
			SG_DONE();
			SG_DEBUG("(%ld iterations)", iterations);
		}

		misclassified=0;
		for (i=0;(i<totdoc);i++) { /* get final statistic */
			if((lin[i]-model->b)*(float64_t)label[i] <= 0.0) 
				misclassified++;
		}

		SG_INFO( "Optimization finished (%ld misclassified, maxdiff=%.8f).\n",
				misclassified,maxdiff); 

		SG_INFO( "obj = %.16f, rho = %.16f\n",get_objective(),model->b);
		if (maxdiff>epsilon)
			SG_WARNING( "maximum violation (%f) exceeds svm_epsilon (%f) due to numerical difficulties\n", maxdiff, epsilon); 

		runtime_end=get_runtime();
		upsupvecnum=0;
		for (i=1;i<model->sv_num;i++)
		{
			if(fabs(model->alpha[i]) >= 
					(learn_parm->svm_cost[model->supvec[i]]-
					 learn_parm->epsilon_a)) 
				upsupvecnum++;
		}
		SG_INFO( "Number of SV: %ld (including %ld at upper bound)\n",
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
  int32_t transductcycle;
  int32_t transduction;
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

  SG_DEBUG("totdoc:%d\n",totdoc);
  chosen = new int32_t[totdoc];
  last_suboptimal_at =new int32_t[totdoc];
  key =new int32_t[totdoc+11];
  selcrit =new float64_t[totdoc];
  selexam =new int32_t[totdoc];
  a_old =new float64_t[totdoc];
  aicache =new float64_t[totdoc];
  working2dnum =new int32_t[totdoc+11];
  active2dnum =new int32_t[totdoc+11];
  qp.opt_ce =new float64_t[learn_parm->svm_maxqpsize];
  qp.opt_ce0 =new float64_t[1];
  qp.opt_g =new float64_t[learn_parm->svm_maxqpsize*learn_parm->svm_maxqpsize];
  qp.opt_g0 =new float64_t[learn_parm->svm_maxqpsize];
  qp.opt_xinit =new float64_t[learn_parm->svm_maxqpsize];
  qp.opt_low=new float64_t[learn_parm->svm_maxqpsize];
  qp.opt_up=new float64_t[learn_parm->svm_maxqpsize];

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

#ifdef CYGWIN
  for (;((iteration<3) || (retrain && (!terminate))||((w_gap>get_weight_epsilon()) && get_mkl_enabled())); iteration++){
#else
  CSignal::clear_cancel();
  for (;((!CSignal::cancel_computations()) && ((iteration<3) || (retrain && (!terminate))||((w_gap>get_weight_epsilon()) && get_mkl_enabled()))); iteration++){
#endif

	  	  
	  if(use_kernel_cache) 
		  kernel->set_time(iteration);  /* for lru cache */
	  
	  if(verbosity>=2) t0=get_runtime();
	  if(verbosity>=3) {
		  SG_DEBUG( "\nSelecting working set... "); 
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
		  SG_INFO( " %ld vectors chosen\n",choosenum); 
	  }
	  
	  if(verbosity>=2) t1=get_runtime();
	  
	  if (use_kernel_cache)
	  {
		  // in case of MKL w/o linadd cache each kernel independently
		  // else if linadd is disabled cache single kernel
		  if ( kernel->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() &&
				  (!((CCombinedKernel*)kernel)->get_append_subkernel_weights()) 
			 )
		  {
			  CCombinedKernel* k      = (CCombinedKernel*) kernel;
			  CKernel* kn = k->get_first_kernel();

			  while (kn)
			  {
				  kn->cache_multiple_kernel_rows(working2dnum, choosenum); 
				  kn = k->get_next_kernel(kn) ;
			  }
		  }
		  else
			  kernel->cache_multiple_kernel_rows(working2dnum, choosenum); 
	  }
	  
	  if(verbosity>=2) t2=get_runtime();
    // !!!
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
		  SG_WARNING( "Relaxing KT-Conditions due to slow progress! Terminating!\n");
	  }
	  
	  noshrink= (get_shrinking_enabled()) ? 0 : 1;
	  if ((!get_mkl_enabled()) && (!retrain) && (inactivenum>0) && ((!learn_parm->skip_final_opt_check) || (kernel->has_property(KP_LINADD) && get_linadd_enabled()))) { 
		  t1=get_runtime();
		  SG_DEBUG( "reactivating inactive examples\n");

		  reactivate_inactive_examples(label,a,shrink_state,lin,c,totdoc,
									   iteration,inconsistent,
									   docs,aicache,
									   maxdiff);
		  reactivated=true;
		  SG_DEBUG("done reactivating inactive examples (maxdiff:%8f eps_crit:%8f orig_eps:%8f)\n", *maxdiff, learn_parm->epsilon_crit, epsilon_crit_org);
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
			  SG_INFO( "restarting optimization as we are - due to shrinking - deviating too much (maxdiff(%f) > eps(%f))\n", *maxdiff, learn_parm->epsilon_crit);
		      retrain=1;
		  }
		  timing_profile->time_shrink+=get_runtime()-t1;
		  if (((verbosity>=1) && (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())))
		     || (verbosity>=2)) {
		      SG_DONE();
		      SG_DEBUG("Number of inactive variables = %ld\n", inactivenum);
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
		  SG_INFO( " => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
					   supvecnum,model->at_upper_bound,(*maxdiff)); 
		  
	  }
	  mymaxdiff=*maxdiff ;

	  //don't shrink w/ mkl
	  if ((!get_mkl_enabled()) && ((iteration % 10) == 0) && (!noshrink))
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

	  SG_ABS_PROGRESS(bestmaxdiff, -CMath::log10(bestmaxdiff), -CMath::log10(worstmaxdiff), -CMath::log10(epsilon), 6);
	  
	  /* Terminate loop */
	  if (max_train_time > 0 && start_time.cur_time_diff() > max_train_time) {
	      terminate = 1;
	      retrain = 0;
	  }
  } /* end of loop */

  SG_DEBUG( "inactive:%d\n", inactivenum);
  if ((!get_mkl_enabled()) && inactivenum && !reactivated)
  {
      SG_DEBUG( "reactivating inactive examples\n");
      reactivate_inactive_examples(label,a,shrink_state,lin,c,totdoc,
    		  iteration,inconsistent,
    		  docs,aicache,
    		  maxdiff);
      SG_DEBUG( "done reactivating inactive examples\n");
      /* Update to new active variables. */
      activenum=compute_index(shrink_state->active,totdoc,active2dnum);
      inactivenum=totdoc-activenum;
      /* reset watchdog */
      bestmaxdiff=(*maxdiff);
      bestmaxdiffiter=iteration;
  }

  criterion=compute_objective_function(a,lin,c,learn_parm->eps,label,totdoc);
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

float64_t CSVMLight::compute_objective_function(
	float64_t *a, float64_t *lin, float64_t *c, float64_t eps, int32_t *label,
	int32_t totdoc)
     /* Return value of objective function. */
     /* Works only relative to the active variables! */
{
  /* calculate value of objective function */
  float64_t criterion=0;

  for (int32_t i=0;i<totdoc;i++)
	  criterion=criterion+(eps-(float64_t)label[i]*c[i])*a[i]+0.5*a[i]*label[i]*lin[i];


  /*float64_t check=0;
  for (int32_t i=0;i<totdoc;i++)
  {
	  check+=a[i]*eps-a[i]*label[i];
	  for (int32_t j=0;j<totdoc;j++)
		  check+= 0.5*a[i]*label[i]*a[j]*label[j]*kernel->kernel(i,j);
  }

  SG_INFO("CLASSIFICATION OBJECTIVE %f vs. CHECK %f (diff %f)\n", criterion, check, criterion-check);
  */

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
     SG_DEBUG( "Running optimizer...");
    }
    /* call the qp-subsolver */
    a_v=optimize_qp(qp,epsilon_crit_target,
		    learn_parm->svm_maxqpsize,
		    &(model->b),  				/* in case the optimizer gives us */
            learn_parm->svm_maxqpsize); /* the threshold for free. otherwise */
                                   		/* b is calculated in calculate_model. */
    if(verbosity>=3) {         
     SG_DONE();
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
	if (parallel.get_num_threads()<=1)
	{
		compute_matrices_for_optimization(docs, label, exclude_from_eq_const, eq_target,
												   chosen, active2dnum, key, a, lin, c, 
												   varnum, totdoc, aicache, qp) ;
	}
#ifndef WIN32
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

		ASSERT(parallel.get_num_threads()>1);
		int32_t *KI=new int32_t[varnum*varnum] ;
		int32_t *KJ=new int32_t[varnum*varnum] ;
		int32_t Knum=0 ;
		float64_t *Kval = new float64_t[varnum*(varnum+1)/2] ;
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
		ASSERT(Knum<=varnum*(varnum+1)/2);

		pthread_t* threads = new pthread_t[parallel.get_num_threads()-1];
		S_THREAD_PARAM_KERNEL* params = new S_THREAD_PARAM_KERNEL[parallel.get_num_threads()-1];
		int32_t step= Knum/parallel.get_num_threads();
		//SG_DEBUG( "\nkernel-step size: %i\n", step) ;
		for (int32_t t=0; t<parallel.get_num_threads()-1; t++)
		{
			params[t].kernel = kernel;
			params[t].start = t*step;
			params[t].end = (t+1)*step;
			params[t].KI=KI ;
			params[t].KJ=KJ ;
			params[t].Kval=Kval ;
			pthread_create(&threads[t], NULL, CSVMLight::compute_kernel_helper, (void*)&params[t]);
		}
		for (i=params[parallel.get_num_threads()-2].end; i<Knum; i++)
			Kval[i]=kernel->kernel(KI[i],KJ[i]) ;

		for (int32_t t=0; t<parallel.get_num_threads()-1; t++)
			pthread_join(threads[t], NULL);

		delete[] params;
		delete[] threads;

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
					SG_DEBUG( "%ld..",i);
				}
			}
		}

		delete[] KI ;
		delete[] KJ ;
		delete[] Kval ;

		for (i=0;i<varnum;i++) {
			/* assure starting at feasible point */
			qp->opt_xinit[i]=a[key[i]];
			/* set linear part of objective function */
			qp->opt_g0[i]=(learn_parm->eps-(float64_t)label[key[i]]*c[key[i]])+qp->opt_g0[i]*(float64_t)label[key[i]];    
		}

		if(verbosity>=3) {
			SG_DONE();
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
			  SG_DEBUG( "%ld..",i);
		  }
	  }
  }

  for (i=0;i<varnum;i++) {
	  /* assure starting at feasible point */
	  qp->opt_xinit[i]=a[key[i]];
	  /* set linear part of objective function */
	  qp->opt_g0[i]=(learn_parm->eps-(float64_t)label[key[i]]*c[key[i]])+qp->opt_g0[i]*(float64_t)label[key[i]];    
  }
  
  if(verbosity>=3) {
	  SG_DONE();
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
   SG_DEBUG( "Calculating model...");
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
     	model->b=((float64_t)label[i]*learn_parm->eps-c[i]+lin[i]); 
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
   SG_DONE();
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
		  target=-(learn_parm->eps-(float64_t)label[i]*c[i]);
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

void CSVMLight::perform_mkl_step(float64_t* beta, float64_t* old_beta, int num_kernels,
		  int32_t* label, int32_t* active2dnum,
		  float64_t* a, float64_t* lin, float64_t* sumw, int32_t& inner_iters)
{
	int32_t num_active_rows=0;
	int32_t num_rows=0;
	float64_t mkl_objective=0;
	int nk = (int) num_kernels; /* calling external lib */
	float64_t suma = 0 ;
	int32_t num = kernel->get_num_vec_rhs();
#ifdef HAVE_LAPACK
	double* alphay  = new double[num];
	
	for (int32_t i=0; i<num; i++)
	{
		alphay[i]=a[i]*label[i];
		suma+=a[i];
	}

	for (int32_t i=0; i<num_kernels; i++)
		sumw[i]=0;
	
	cblas_dgemv(CblasColMajor, CblasNoTrans, num_kernels, (int) num, 0.5, (double*) W,
		num_kernels, alphay, 1, 1.0, (double*) sumw, 1);

	mkl_objective=-suma;
	for (int32_t i=0; i<num_kernels; i++)
		mkl_objective+=old_beta[i]*sumw[i];

	delete[] alphay;
#else
	for (int32_t i=0; i<num; i++)
		suma += a[i];

	mkl_objective=-suma;
	for (int32_t d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for (int32_t i=0; i<num; i++)
			sumw[d] += a[i]*(0.5*label[i]*W[i*num_kernels+d]);
		mkl_objective   += old_beta[d]*sumw[d];
	}
#endif
	
	count++ ;

	w_gap = CMath::abs(1-rho/mkl_objective) ;
	if( (w_gap >= 0.9999*get_weight_epsilon()) || (get_solver_type()==ST_INTERNAL && mkl_norm>1) )
	{
		if ( mkl_norm == 1)
		{
#ifdef USE_CPLEX
			if (get_solver_type()==ST_CPLEX || get_solver_type()==ST_AUTO)
				rho=compute_optimal_betas_via_cplex(beta, old_beta, num_kernels, sumw, suma, inner_iters);
			else
				rho=compute_optimal_betas_via_glpk(beta, old_beta, num_kernels, sumw, suma, inner_iters);
#else
			if (get_solver_type()==ST_GLPK || get_solver_type()==ST_AUTO)
				rho=compute_optimal_betas_via_glpk(beta, old_beta, num_kernels, sumw, suma, inner_iters);
			else
				rho=compute_optimal_betas_via_cplex(beta, old_beta, num_kernels, sumw, suma, inner_iters);
#endif
		}
		else
		{
			if (get_solver_type()==ST_CPLEX || get_solver_type()==ST_AUTO) {
				rho=compute_optimal_betas_via_cplex(beta, old_beta, num_kernels, sumw, suma, inner_iters);
      } else {
				//rho=compute_optimal_betas_analytically(beta, old_beta, num_kernels, sumw, suma, mkl_objective);
				//rho=compute_optimal_betas_gradient(beta, old_beta, num_kernels, sumw, suma, mkl_objective);
				rho=compute_optimal_betas_newton(beta, old_beta, num_kernels, sumw, suma, mkl_objective);
      }
		}

		// set weights, store new rho and compute new w gap
		kernel->set_subkernel_weights(beta, num_kernels) ;
		w_gap = CMath::abs(1-rho/mkl_objective) ;
	}

	// update lin
#ifdef HAVE_LAPACK
	cblas_dgemv(CblasColMajor, CblasTrans, nk, (int) num, 1.0, (double*) W,
		nk, (double*) old_beta, 1, 0.0, (double*) lin, 1);
#else
	for (int32_t i=0; i<num; i++)
		lin[i]=0 ;
	for (int32_t d=0; d<num_kernels; d++)
		if (old_beta[d]!=0)
			for (int32_t i=0; i<num; i++)
				lin[i] += old_beta[d]*W[i*num_kernels+d] ;
#endif
	
	// count actives
	int32_t jj;
	for (jj=0;active2dnum[jj]>=0;jj++);
	
	if (count%10==0)
	{
		int32_t start_row = 1 ;
		if (C_mkl!=0.0)
			start_row+=2*(num_kernels-1);
		SG_DEBUG("%i. OBJ: %f  RHO: %f  wgap=%f agap=%f (activeset=%i; active rows=%i/%i; inner_iters=%d)\n", count, mkl_objective,rho,w_gap,mymaxdiff,jj,num_active_rows,num_rows-start_row, inner_iters);
	}
}

float64_t CSVMLight::compute_optimal_betas_analytically(float64_t* beta,
		float64_t* old_beta, int32_t num_kernels,
		const float64_t* sumw, float64_t suma,
    float64_t mkl_objective)
{
	SG_DEBUG("MKL via ANALYTICAL\n");

	const float64_t r = 1.0 / ( mkl_norm - 1.0 );
	float64_t Z;
	float64_t obj;

  /*
  obj = -suma;
	for (int32_t d=0; d<num_kernels; d++) {
		obj += old_beta[d] * (sumw[d]);
  }
	SG_PRINT( "OBJ_old = %f / %f\n", obj, mkl_objective );
  */
	SG_PRINT( "OBJ_old = %f\n", mkl_objective );

	Z = 0.0;
	for (int32_t n=0; n<num_kernels; n++ ) {
		beta[n] = CMath::pow( sumw[n], r );
		Z += CMath::pow( beta[n], mkl_norm );
	}

	Z = CMath::pow( Z, -1.0/mkl_norm );
	for( int32_t n=0; n<num_kernels; n++ ) {
		beta[n] *= Z;
  }
	CMath::display_vector( beta, num_kernels, "new_beta " );

	Z = 0.0;
	for (int32_t n=0; n<num_kernels; n++ ) {
		beta[n] = beta[n] * old_beta[n];
		Z += CMath::pow( beta[n], mkl_norm );
	}
	Z = CMath::pow( Z, -1.0/mkl_norm );
	for( int32_t n=0; n<num_kernels; n++ ) {
		beta[n] *= Z;
  }

	// CMath::display_vector(beta, num_kernels, "beta_alex");
	SG_PRINT("Z_alex = %e\n", Z);

  /*
	// compute in log space
	Z = CMath::log(0.0);
	for (int32_t n=0; n<num_kernels; n++ )
	{
		ASSERT(sumw[n]>=0);
		beta[n] = CMath::log(sumw[n])*r;
		Z = CMath::logarithmic_sum(Z, beta[n]*mkl_norm);
	}

	Z *= -1.0/mkl_norm;
	for (int32_t n=0; n<num_kernels; n++ )
		beta[n] = CMath::exp(beta[n]+Z);
  */

	CMath::display_vector( old_beta, num_kernels, "old_beta " );
	CMath::display_vector( beta,     num_kernels, "beta     " );
	//CMath::display_vector(beta, num_kernels, "beta_log");
	//SG_PRINT("Z_log=%f\n", Z);
	//for (int32_t i=0; i<num_kernels; i++)
		//beta[i]=(beta[i]+old_beta[i])/2;

	//CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

  obj = -suma;
	for (int32_t d=0; d<num_kernels; d++) {
		obj += beta[d] * (sumw[d]);
  }
	SG_PRINT( "OBJ = %f\n", obj );
	return obj;
}

/*
float64_t CSVMLight::compute_optimal_betas_gradient(float64_t* beta,
		float64_t* old_beta, int32_t num_kernels,
		const float64_t* sumw, float64_t suma,
    float64_t mkl_objective)
{
	SG_DEBUG("MKL via GRADIENT\n");

	const float64_t r = mkl_norm / ( mkl_norm - 1.0 );
	float64_t Z;
	float64_t obj;
	float64_t gamma;
	float64_t maxstep;
  int32_t p;

	SG_PRINT( "OBJ_old = %f\n", mkl_objective );

	gamma = 0.0;
	for( p=0; p<num_kernels; ++p ) {
		gamma += CMath::pow( sumw[p], r );
	}
  gamma = CMath::pow( gamma, 1.0/r ) / mkl_norm;

  // compute gradient (stored in "beta")
  maxstep = CMath::INFTY;
  maxstep = 0.0;
	for( p=0; p<num_kernels; ++p ) {
    ASSERT( 0.0 <= old_beta[p] && old_beta[p] <= 1.0 );
		beta[p] = gamma * mkl_norm * CMath::pow( old_beta[p], mkl_norm-1 ) - sumw[p];
    const float64_t step = ( beta[p] > 0.0 ) ? old_beta[p]/beta[p] : (old_beta[p]-1.0)/beta[p];
    ASSERT( step >= 0.0 );
    if( step > maxstep ) {
      maxstep = step;
    }
  }
  ASSERT( maxstep > 0.0 );

  // make gradient step
	Z = 0.0;
	for( p=0; p<num_kernels; ++p ) {
		beta[p] = old_beta[p] - 0.5*maxstep*beta[p];
    //ASSERT( 0.0 <= beta[p] && beta[p] <= 1.0 );
    if( beta[p] < 0.0 ) {
      beta[p] = 0.0;
    }
		Z += CMath::pow( beta[p], mkl_norm );
	}
	Z = CMath::pow( Z, -1.0/mkl_norm );
	for( p=0; p<num_kernels; ++p ) {
		beta[p] *= Z;
  }

	// CMath::display_vector(beta, num_kernels, "beta_alex");
	SG_PRINT("Z_alex = %e\n", Z);

	CMath::display_vector( old_beta, num_kernels, "old_beta " );
	CMath::display_vector( beta,     num_kernels, "beta     " );
	//CMath::display_vector(beta, num_kernels, "beta_log");
	//SG_PRINT("Z_log=%f\n", Z);
	//for (int32_t i=0; i<num_kernels; i++)
		//beta[i]=(beta[i]+old_beta[i])/2;

	//CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

  obj = -suma;
	for( p=0; p<num_kernels; ++p ) {
		obj += beta[p] * (sumw[p]);
  }
	SG_PRINT( "OBJ = %f\n", obj );
	return obj;
}
*/

/*
float64_t CSVMLight::compute_optimal_betas_gradient(float64_t* beta,
		float64_t* old_beta, int32_t num_kernels,
		const float64_t* sumw, float64_t suma,
    float64_t mkl_objective)
{
	SG_DEBUG("MKL via GRADIENT-EXP\n");

	const float64_t r = mkl_norm / ( mkl_norm - 1.0 );
	float64_t Z;
	float64_t obj;
	float64_t gamma;
  int32_t p;

	SG_PRINT( "OBJ_old = %f\n", mkl_objective );

	gamma = 0.0;
	for( p=0; p<num_kernels; ++p ) {
		gamma += CMath::pow( sumw[p], r );
	}
  gamma = CMath::pow( gamma, 1.0/r ) / mkl_norm;

  // compute gradient (stored in "beta")
	for( p=0; p<num_kernels; ++p ) {
    ASSERT( 0.0 <= old_beta[p] && old_beta[p] <= 1.0 );
		beta[p] = gamma * mkl_norm * CMath::pow( old_beta[p], mkl_norm-1 ) - sumw[p];
  }
	for( p=0; p<num_kernels; ++p ) {
    beta[p] *= ( mkl_norm - 1.0 ) * ( mkl_norm - 1.0 );
  }
	CMath::display_vector( beta, num_kernels, "grad" );

  // make gradient step in log-beta space
  Z = 0.0;
	for( p=0; p<num_kernels; ++p ) {
		beta[p] = CMath::log(old_beta[p]) - beta[p] / old_beta[p];
		beta[p] = CMath::exp( beta[p] );
    if( beta[p] != beta[p] || beta[p] == CMath::INFTY ) {
      beta[p] = 0.0;
    }
    ASSERT( beta[p] == beta[p] );
		Z += CMath::pow( beta[p], mkl_norm );
	}
	Z = CMath::pow( Z, -1.0/mkl_norm );
	CMath::display_vector( beta, num_kernels, "pre-beta " );
	for( p=0; p<num_kernels; ++p ) {
		beta[p] *= Z;
  }

	// CMath::display_vector(beta, num_kernels, "beta_alex");
	SG_PRINT("Z_alex = %e\n", Z);

	CMath::display_vector( old_beta, num_kernels, "old_beta " );
	CMath::display_vector( beta,     num_kernels, "beta     " );

  obj = -suma;
	for( p=0; p<num_kernels; ++p ) {
		obj += beta[p] * (sumw[p]);
  }
	SG_PRINT( "OBJ = %f\n", obj );
	return obj;
}
*/

float64_t CSVMLight::compute_optimal_betas_newton(float64_t* beta,
		float64_t* old_beta, int32_t num_kernels,
		const float64_t* sumw, float64_t suma,
    float64_t mkl_objective)
{
	SG_DEBUG("MKL via NEWTON\n");

	const float64_t r = mkl_norm / ( mkl_norm - 1.0 );
	float64_t Z;
	float64_t obj;
	float64_t gamma;
  int32_t p;

	//SG_PRINT( "OBJ_old = %f\n", mkl_objective );

	gamma = 0.0;
	for( p=0; p<num_kernels; ++p ) {
		gamma += CMath::pow( sumw[p], r );
	}
  gamma = CMath::pow( gamma, 1.0/r ) / mkl_norm;

  /*
  // compute gradient (stored in "beta")
	for( p=0; p<num_kernels; ++p ) {
    ASSERT( 0.0 <= old_beta[p] && old_beta[p] <= 1.0 );
		beta[p] = gamma * mkl_norm * CMath::pow( old_beta[p], mkl_norm-1.0 ) - sumw[p];
  }

  // compute Newton step (Hessian is diagonal)
  const float64_t gqq1 = gamma * mkl_norm * (mkl_norm-1.0);
  Z = 0.0;
	for( p=0; p<num_kernels; ++p ) {
		beta[p] /= 2.0*sumw[p]/old_beta[p] + gqq1*CMath::pow(old_beta[p],mkl_norm-2.0);
    Z += beta[p] * beta[p];
  }
	//CMath::display_vector( beta, num_kernels, "newton   " );
	//SG_PRINT( "Newton step size = %e\n", Z );
  */

  //float64_t* myLastBeta = new float64_t[ num_kernels ];
  const int nofNewtonSteps = 1;
  int i;
  if( nofNewtonSteps > 1 ) {
    SG_PRINT( "performing %d Newton steps.\n", nofNewtonSteps );
  }
  for( i = 0; i < nofNewtonSteps; ++i ) {

    if( i != 0 ) {
      for( p=0; p<num_kernels; ++p ) {
        old_beta[p] = beta[p];
      }
    }

    // compute Newton step (stored in "beta") (Hessian is diagonal)
    const float64_t gqq1 = gamma * mkl_norm * (mkl_norm-1.0);
    Z = 0.0;
    for( p=0; p<num_kernels; ++p ) {
      ASSERT( 0.0 <= old_beta[p] && old_beta[p] <= 1.0 );
      beta[p] = ( gamma*mkl_norm*CMath::pow(old_beta[p],mkl_norm) - sumw[p]*old_beta[p] )
        / ( 2.0*sumw[p] + gqq1*CMath::pow(old_beta[p],mkl_norm-1.0) );
      Z += beta[p] * beta[p];
    }
    //CMath::display_vector( beta, num_kernels, "newton   " );
    //SG_PRINT( "Newton step size = %e\n", Z );

    // perform Newton step
    Z = 0.0;
    for( p=0; p<num_kernels; ++p ) {
      beta[p] = old_beta[p] - beta[p];
      //ASSERT( 0.0 <= beta[p] && beta[p] <= 1.0 );
      if( beta[p] < 0.0 ) {
        beta[p] = 0.0;
      }
      Z += CMath::pow( beta[p], mkl_norm );
    }
    Z = CMath::pow( Z, -1.0/mkl_norm );
    for( p=0; p<num_kernels; ++p ) {
      beta[p] *= Z;
    }

  }

	// CMath::display_vector(beta, num_kernels, "beta_alex");
	//SG_PRINT("Z_alex = %e\n", Z);
	//CMath::display_vector( old_beta, num_kernels, "old_beta " );
	//CMath::display_vector( beta,     num_kernels, "beta     " );
	//CMath::display_vector(beta, num_kernels, "beta_log");
	//SG_PRINT("Z_log=%f\n", Z);
	//for (int32_t i=0; i<num_kernels; i++)
		//beta[i]=(beta[i]+old_beta[i])/2;

	//CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

  obj = -suma;
	for( p=0; p<num_kernels; ++p ) {
		obj += beta[p] * (sumw[p]);
  }
	//SG_PRINT( "OBJ = %f\n", obj );
	return obj;
}

float64_t CSVMLight::compute_optimal_betas_via_cplex(float64_t* x, float64_t* old_beta, int32_t num_kernels,
		  const float64_t* sumw, float64_t suma, int32_t& inner_iters)
{
	SG_DEBUG("MKL via CPLEX\n");

#ifdef USE_CPLEX
	if (!lp_initialized)
	{
		SG_INFO( "creating LP\n") ;

		int32_t NUMCOLS = 2*num_kernels + 1 ;
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
			SG_ERROR( "%s", errmsg);
		}

		// add constraint sum(w)=1;
		SG_INFO( "adding the first row\n");
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

			for (int32_t i=0; i<num_kernels; i++)
			{
				initial_rmatind[i]=i ;
				initial_rmatval[i]=1 ;
			}
			initial_rmatind[num_kernels]=2*num_kernels ;
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
			SG_ERROR( "Failed to add the first row.\n");

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
					SG_ERROR( "Failed to add a smothness row (1).\n");

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
					SG_ERROR( "Failed to add a smothness row (2).\n");
			}
		}
	}

	{ // add the new row
		//SG_INFO( "add the new row\n") ;

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
			SG_ERROR( "Failed to add the new row.\n");
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
			float64_t* beta=new float64_t[2*num_kernels+1];
			float64_t objval_old=1e-8; //some value to cause the loop to not terminate yet
			for (int32_t i=0; i<num_kernels; i++)
				beta[i]=old_beta[i];
			for (int32_t i=num_kernels; i<2*num_kernels+1; i++)
				beta[i]=0;

			while (true)
			{
				//int rows=CPXgetnumrows(env, lp_cplex);
				//int cols=CPXgetnumcols(env, lp_cplex);
				//SG_PRINT("rows:%d, cols:%d (kernel:%d)\n", rows, cols, num_kernels);
				CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

				set_qnorm_constraints(beta, num_kernels);

				status = CPXbaropt(env, lp_cplex);
				if ( status ) 
					SG_ERROR( "Failed to optimize Problem.\n");

				int solstat=0;
				double objval=0;
				status=CPXsolution(env, lp_cplex, &solstat, &objval,
						(double*) beta, NULL, NULL, NULL);

				if ( status )
				{
					CMath::display_vector(beta, num_kernels, "beta");
					SG_ERROR( "Failed to obtain solution.\n");
				}

				CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

				//SG_PRINT("[%d] %f (%f)\n", inner_iters, objval, objval_old);
				if ((1-abs(objval/objval_old) < 0.1*weight_epsilon)) // && (inner_iters>2))
					break;

				objval_old=objval;

				inner_iters++;
			}
			delete[] beta;
		}

		if ( status ) 
			SG_ERROR( "Failed to optimize Problem.\n");

		// obtain solution
		int32_t cur_numrows=(int32_t) CPXgetnumrows(env, lp_cplex);
		int32_t cur_numcols=(int32_t) CPXgetnumcols(env, lp_cplex);
		int32_t num_rows=cur_numrows;
		ASSERT(cur_numcols<=2*num_kernels+1);

		float64_t* slack=new float64_t[cur_numrows];
		float64_t* pi=NULL;
		if (use_mkl==1)
			pi=new float64_t[cur_numrows];

		if (x==NULL || slack==NULL || pi==NULL)
		{
			status = CPXERR_NO_MEMORY;
			SG_ERROR( "Could not allocate memory for solution.\n");
		}

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
			SG_ERROR( "Failed to obtain solution.\n");

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
				//SG_INFO( "-%i(%i,%i)",max_idx,start_row,num_rows) ;
				status = CPXdelrows (env, lp_cplex, max_idx, max_idx) ;
				if ( status ) 
					SG_ERROR( "Failed to remove an old row.\n");
			}

			//CMath::display_vector(x, num_kernels, "beta");

			rho = -x[2*num_kernels] ;
			delete[] pi ;
			delete[] slack ;

		}
		else
		{
			/* then something is wrong and we rather 
			stop sooner than later */
			rho = 1 ;
		}
	}
#else
	SG_ERROR("Cplex not enabled at compile time\n");
#endif
	return rho;
}

float64_t CSVMLight::compute_optimal_betas_via_glpk(float64_t* beta, float64_t* old_beta,
		int num_kernels, const float64_t* sumw, float64_t suma, int32_t& inner_iters)
{
	SG_DEBUG("MKL via GLPK\n");
	float64_t obj=-suma;
#ifdef USE_GLPK
	int32_t NUMCOLS = 2*num_kernels + 1 ;
	if (!lp_initialized)
	{
		SG_INFO( "initializing lp...");
		//set obj function.
		lpx_add_cols(lp_glpk, NUMCOLS);
		for (int i=1; i<=2*num_kernels; i++)
		{
			lpx_set_obj_coef(lp_glpk, i, 0);
			lpx_set_col_bnds(lp_glpk, i, LPX_DB, 0, 1);
		}
		for (int i=num_kernels+1; i<=2*num_kernels; i++)
		{
			lpx_set_obj_coef(lp_glpk, i, C_mkl);
		}
		lpx_set_obj_coef(lp_glpk, NUMCOLS, 1);
		lpx_set_col_bnds(lp_glpk, NUMCOLS, LPX_FR, -CMath::INFTY, CMath::INFTY);

		//add first row. sum[w]=1
		int row_index = lpx_add_rows(lp_glpk, 1);
		int ind[num_kernels+2];
		float64_t val[num_kernels+1];
		for (int i=1; i<=num_kernels; i++)
		{
			ind[i] = i;
			val[i] = 1;
		}
		//			ind[num_kernels+1] = NUMCOLS;
		//			val[num_kernels+1] = 0;
		lpx_set_mat_row(lp_glpk, row_index, num_kernels, ind, val);
		lpx_set_row_bnds(lp_glpk, row_index, LPX_FX, 1, 1);

		lp_initialized = true;

		if (C_mkl!=0.0)
		{
			for (int32_t q=1; q<num_kernels; q++)
			{
				int mat_ind[4];
				float64_t mat_val[4];
				int mat_row_index = lpx_add_rows(lp_glpk, 2);
				mat_ind[1] = q;
				mat_val[1] = 1;
				mat_ind[2] = q+1; 
				mat_val[2] = -1;
				mat_ind[3] = num_kernels+q;
				mat_val[3] = -1;
				lpx_set_mat_row(lp_glpk, mat_row_index, 3, mat_ind, mat_val);
				lpx_set_row_bnds(lp_glpk, mat_row_index, LPX_UP, 0, 0);
				mat_val[1] = -1; 
				mat_val[2] = 1;
				lpx_set_mat_row(lp_glpk, mat_row_index+1, 3, mat_ind, mat_val);
				lpx_set_row_bnds(lp_glpk, mat_row_index+1, LPX_UP, 0, 0);
			}
		}
	}

	int ind[num_kernels+2];
	float64_t val[num_kernels+2];
	int row_index = lpx_add_rows(lp_glpk, 1);
	for (int32_t i=1; i<=num_kernels; i++)
	{
		ind[i] = i;
		val[i] = -sumw[i-1];
	}
	ind[num_kernels+1] = 2*num_kernels+1;
	val[num_kernels+1] = -1;
	lpx_set_mat_row(lp_glpk, row_index, num_kernels+1, ind, val);
	lpx_set_row_bnds(lp_glpk, row_index, LPX_UP, 0, 0);

	//optimize
	lpx_simplex(lp_glpk);
	check_lpx_status(lp_glpk);	
	int32_t cur_numrows = lpx_get_num_rows(lp_glpk);
	int32_t cur_numcols = lpx_get_num_cols(lp_glpk);
	int32_t num_rows=cur_numrows;
	ASSERT(cur_numcols<=2*num_kernels+1);

	float64_t *row_primal = new float64_t[cur_numrows];
	float64_t *row_dual = new float64_t[cur_numrows];

	for (int i=0; i<cur_numrows; i++)
	{
		row_primal[i] = lpx_get_row_prim(lp_glpk, i+1);
		row_dual[i] = lpx_get_row_dual(lp_glpk, i+1);
	}
	for (int i=0; i<cur_numcols; i++)
	{
		beta[i] = lpx_get_col_prim(lp_glpk, i+1);
	}
	//CMath::display_vector(beta, cur_numcols, "beta");

	bool res = check_lpx_status(lp_glpk);
	if (!res)
		SG_ERROR("Failed to obtain solution.\n");

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
			lpx_del_rows(lp_glpk, 1, del_rows);
		}

		for (int32_t d=0; d<num_kernels; d++)
			obj   += beta[d]*(sumw[d]);
		return obj;

		delete[] row_dual;
		delete[] row_primal;
	}
	else
	{
		/* then something is wrong and we rather 
		   stop sooner than later */
		obj = 1 ;
	}
#else
	SG_ERROR("Glpk not enabled at compile time\n");
#endif
	return obj;
}


void CSVMLight::update_linear_component_mkl(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache)
{
	int inner_iters=0;
	int32_t num = kernel->get_num_vec_rhs();
	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	const float64_t* beta_const   = kernel->get_subkernel_weights(num_weights);
	float64_t* old_beta =  CMath::clone_vector(beta_const, num_weights);
	// large enough buffer for cplex + smoothness constraints
	float64_t* beta = new float64_t[2*num_kernels+1];

	ASSERT(num_weights==num_kernels);
	CMath::scale_vector(1/CMath::qnorm(old_beta, num_kernels, mkl_norm), old_beta, num_kernels); //q-norm = 1

	float64_t* sumw=new float64_t[num_kernels];

	if ((kernel->get_kernel_type()==K_COMBINED) && 
			 (!((CCombinedKernel*)kernel)->get_append_subkernel_weights()))// for combined kernel
	{
		CCombinedKernel* k      = (CCombinedKernel*) kernel;
		CKernel* kn = k->get_first_kernel() ;
		int32_t n = 0, i, j ;
		
		while (kn!=NULL)
		{
			for (i=0;i<num;i++) 
			{
				if(a[i] != a_old[i]) 
				{
					kn->get_kernel_row(i,NULL,aicache, true);
					for (j=0;j<num;j++)
						W[j*num_kernels+n]+=(a[i]-a_old[i])*aicache[j]*(float64_t)label[i];
				}
			}
			kn = k->get_next_kernel(kn) ;
			n++ ;
		}
	}
	else // hope the kernel is fast ...
	{
		float64_t* w_backup = new float64_t[num_kernels] ;
		float64_t* w1 = new float64_t[num_kernels] ;
		
		// backup and set to zero
		for (int32_t i=0; i<num_kernels; i++)
		{
			w_backup[i] = old_beta[i] ;
			w1[i]=0.0 ; 
		}
		for (int32_t n=0; n<num_kernels; n++)
		{
			w1[n]=1.0 ;
			kernel->set_subkernel_weights(w1, num_weights) ;
		
			for (int32_t i=0;i<num;i++) 
			{
				if(a[i] != a_old[i]) 
				{
					for (int32_t j=0;j<num;j++) 
						W[j*num_kernels+n]+=(a[i]-a_old[i])*kernel->kernel(i,j)*(float64_t)label[i];
				}
			}
			w1[n]=0.0 ;
		}

		// restore old weights
		kernel->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;
	}
	
	perform_mkl_step(beta, old_beta, num_kernels, label, active2dnum,
			a, lin, sumw, inner_iters);
	
	delete[] sumw;
	delete[] old_beta;
	delete[] beta;
}


void CSVMLight::update_linear_component_mkl_linadd(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache)
{
	int inner_iters=0;

	// kernel with LP_LINADD property is assumed to have 
	// compute_by_subkernel functions
	int32_t num = kernel->get_num_vec_rhs();
	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	const float64_t* beta_const   = kernel->get_subkernel_weights(num_weights);
	float64_t* old_beta =  CMath::clone_vector(beta_const, num_weights);
	// large enough buffer for cplex + smoothness constraints
	float64_t* beta = new float64_t[2*num_kernels+1];

	ASSERT(num_weights==num_kernels);
	CMath::scale_vector(1/CMath::qnorm(old_beta, num_kernels, mkl_norm), old_beta, num_kernels); //q-norm = 1

	float64_t* sumw = new float64_t[num_kernels];
	float64_t* w_backup = new float64_t[num_kernels] ;
	float64_t* w1 = new float64_t[num_kernels] ;

	// backup and set to one
	for (int32_t i=0; i<num_kernels; i++)
	{
		w_backup[i] = old_beta[i] ;
		w1[i]=1.0 ; 
	}
	// set the kernel weights
	kernel->set_subkernel_weights(w1, num_weights) ;

	// create normal update (with changed alphas only)
	kernel->clear_normal();
	for (int32_t ii=0, i=0;(i=working2dnum[ii])>=0;ii++) {
		if(a[i] != a_old[i]) {
			kernel->add_to_normal(docs[i], (a[i]-a_old[i])*(float64_t)label[i]);
		}
	}

	if (parallel.get_num_threads() < 2)
	{
		// determine contributions of different kernels
		for (int32_t i=0; i<num; i++)
			kernel->compute_by_subkernel(i,&W[i*num_kernels]);
	}
#ifndef WIN32
	else
	{
		pthread_t* threads = new pthread_t[parallel.get_num_threads()-1];
		S_THREAD_PARAM* params = new S_THREAD_PARAM[parallel.get_num_threads()-1];
		int32_t step= num/parallel.get_num_threads();

		for (int32_t t=0; t<parallel.get_num_threads()-1; t++)
		{
			params[t].kernel = kernel;
			params[t].W = W;
			params[t].start = t*step;
			params[t].end = (t+1)*step;
			pthread_create(&threads[t], NULL, CSVMLight::update_linear_component_mkl_linadd_helper, (void*)&params[t]);
		}

		for (int32_t i=params[parallel.get_num_threads()-2].end; i<num; i++)
			kernel->compute_by_subkernel(i,&W[i*num_kernels]);

		for (int32_t t=0; t<parallel.get_num_threads()-1; t++)
			pthread_join(threads[t], NULL);

		delete[] params;
		delete[] threads;
	}
#endif

	// restore old weights
	kernel->set_subkernel_weights(w_backup,num_weights);

	delete[] w_backup;
	delete[] w1;

	perform_mkl_step(beta, old_beta, num_kernels, label, active2dnum,
			a, lin, sumw, inner_iters);
	
	delete[] sumw;
	delete[] old_beta;
	delete[] beta;
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
		if (kernel->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() ) 
		{
			update_linear_component_mkl_linadd(docs, label, active2dnum, a, a_old, working2dnum, 
											   totdoc,	lin, aicache) ;
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
				if (parallel.get_num_threads() < 2)
				{
					for (jj=0;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=kernel->compute_optimized(docs[j]);
					}
				}
#ifndef WIN32
				else
				{
					int32_t num_elem = 0 ;
					for (jj=0;(j=active2dnum[jj])>=0;jj++) num_elem++ ;

					pthread_t* threads = new pthread_t[parallel.get_num_threads()-1] ;
					S_THREAD_PARAM* params = new S_THREAD_PARAM[parallel.get_num_threads()-1] ;
					int32_t start = 0 ;
					int32_t step = num_elem/parallel.get_num_threads();
					int32_t end = step ;

					for (int32_t t=0; t<parallel.get_num_threads()-1; t++)
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

					for (jj=params[parallel.get_num_threads()-2].end;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=kernel->compute_optimized(docs[j]);
					}
					void* ret;
					for (int32_t t=0; t<parallel.get_num_threads()-1; t++)
						pthread_join(threads[t], &ret) ;

					delete[] params;
					delete[] threads;
				}
#endif
			}
		}
	}
	else 
	{
		if (kernel->has_property(KP_KERNCOMBINATION) && get_mkl_enabled() ) 
		{
			update_linear_component_mkl(docs, label, active2dnum, a, a_old, working2dnum, 
										totdoc,	lin, aicache) ;
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
			selcrit[activedoc]=(float64_t)label[j]*(learn_parm->eps-(float64_t)label[j]*c[j]+(float64_t)label[j]*lin[j]);
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
			selcrit[activedoc]=-(float64_t)label[j]*(learn_parm->eps-(float64_t)label[j]*c[j]+(float64_t)label[j]*lin[j]);
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
  shrink_state->active = new int32_t[totdoc];
  shrink_state->inactive_since = new int32_t[totdoc];
  shrink_state->a_history = new float64_t*[maxhistory];
  shrink_state->maxhistory=maxhistory;
  shrink_state->last_lin = new float64_t[totdoc];
  shrink_state->last_a = new float64_t[totdoc];

  for (i=0;i<totdoc;i++) { 
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
		  SG_INFO( " Shrinking...");
	  }

	  if (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())) { /*  non-linear case save alphas */
	 
		  a_old=new float64_t[totdoc];
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
		  SG_DONE();
		  SG_DEBUG("Number of inactive variables = %ld\n", totdoc-activenum);
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

#ifdef USE_CPLEX
void CSVMLight::set_qnorm_constraints(float64_t* beta, int32_t num_kernels)
{
	ASSERT(num_kernels>0);

	float64_t* grad_beta=new float64_t[num_kernels];
	float64_t* hess_beta=new float64_t[num_kernels+1];
	float64_t* lin_term=new float64_t[num_kernels+1];
	int* ind=new int[num_kernels+1];

	//CMath::display_vector(beta, num_kernels, "beta");
	double const_term = 1-CMath::qsq(beta, num_kernels, mkl_norm);
	//SG_PRINT("const=%f\n", const_term);
	ASSERT(CMath::fequal(const_term, 0.0));

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
		ASSERT(!status);
	}

	status = CPXaddqconstr (env, lp_cplex, num_kernels+1, num_kernels+1, const_term, 'L', ind, lin_term,
			ind, ind, hess_beta, NULL);
	ASSERT(!status);

	//CPXwriteprob (env, lp_cplex, "prob.lp", NULL);
	//CPXqpwrite (env, lp_cplex, "prob.qp");

	delete[] grad_beta;
	delete[] hess_beta;
	delete[] lin_term;
	delete[] ind;
}
#endif // USE_CPLEX

void* CSVMLight::reactivate_inactive_examples_vanilla_helper(void* p)
{
	S_THREAD_PARAM_REACTIVATE_VANILLA* params = (S_THREAD_PARAM_REACTIVATE_VANILLA*) p;
	ASSERT(params);
	ASSERT(params->kernel);
	ASSERT(params->lin);
	ASSERT(params->aicache);
	ASSERT(params->a);
	ASSERT(params->a_old);
	ASSERT(params->changed2dnum);
	ASSERT(params->inactive2dnum);
	ASSERT(params->label);

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
		ASSERT(i>=0);

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
		  SG_DEBUG( " clear normal - linadd\n");
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
			  int32_t num_threads=parallel.get_num_threads();
			  ASSERT(num_threads>0);
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
#ifndef WIN32
			  else
			  {
				  pthread_t* threads = new pthread_t[num_threads-1];
				  S_THREAD_PARAM_REACTIVATE_LINADD* params = new S_THREAD_PARAM_REACTIVATE_LINADD[num_threads];
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

				  delete[] threads;
				  delete[] params;
			  }
#endif

		  }
	  }
	  else 
	  {
		  float64_t *alphas = new float64_t[totdoc] ;
		  int32_t *idx = new int32_t[totdoc] ;
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
			  int32_t* inactive_idx=new int32_t[totdoc]; // infact we only need a subset 

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
				  float64_t* dest = new float64_t[num_inactive];
				  memset(dest, 0, sizeof(float64_t)*num_inactive);

				  kernel->compute_batch(num_inactive, inactive_idx, dest, num_suppvec, idx, alphas);

				  j=0;
				  for (i=0;i<totdoc;i++) {
					  if(!shrink_state->active[i]) {
						  lin[i] = shrink_state->last_lin[i] + dest[j++] ;
					  }
					  shrink_state->last_lin[i]=lin[i];
				  }

				  delete[] dest;
			  }
			  else
			  {
				  for (i=0;i<totdoc;i++)
					  shrink_state->last_lin[i]=lin[i];
			  }
			  delete[] inactive_idx;
		  }
		  delete[] alphas;
		  delete[] idx;
	  }

	  kernel->delete_optimization();
  }
  else 
  {
	  changed=new int32_t[totdoc];
	  changed2dnum=new int32_t[totdoc+11];
	  inactive=new int32_t[totdoc];
	  inactive2dnum=new int32_t[totdoc+11];
	  for (t=shrink_state->deactnum-1;(t>=0) && shrink_state->a_history[t];t--)
	  {
		  if(verbosity>=2) {
			  SG_INFO( "%ld..",t);
		  }
		  a_old=shrink_state->a_history[t];    
		  for (i=0;i<totdoc;i++) {
			  inactive[i]=((!shrink_state->active[i]) 
						   && (shrink_state->inactive_since[i] == t));
			  changed[i]= (a[i] != a_old[i]);
		  }
		  compute_index(inactive,totdoc,inactive2dnum);
		  compute_index(changed,totdoc,changed2dnum);


		  int32_t num_threads=parallel.get_num_threads();
		  ASSERT(num_threads>0);

		  if (num_threads < 2)
		  {
			  for (ii=0;(i=changed2dnum[ii])>=0;ii++) {
				  kernel->get_kernel_row(i,inactive2dnum,aicache);
				  for (jj=0;(j=inactive2dnum[jj])>=0;jj++)
					  lin[j]+=(a[i]-a_old[i])*aicache[j]*(float64_t)label[i];
			  }
		  }
#ifndef WIN32
		  else
		  {
			  //find number of the changed ones
			  int32_t num_changed=0;
			  for (ii=0;changed2dnum[ii]>=0;ii++)
				  num_changed++;
			
			  if (num_changed>0)
			  {
				  pthread_t* threads= new pthread_t[num_threads-1];
				  S_THREAD_PARAM_REACTIVATE_VANILLA* params = new S_THREAD_PARAM_REACTIVATE_VANILLA[num_threads];
				  int32_t step= num_changed/num_threads;

				  // alloc num_threads many tmp buffers
				  float64_t* tmp_lin=new float64_t[totdoc*num_threads];
				  memset(tmp_lin, 0, sizeof(float64_t)*((size_t) totdoc)*num_threads);
				  float64_t* tmp_aicache=new float64_t[totdoc*num_threads];
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

				  delete[] tmp_lin;
				  delete[] tmp_aicache;
				  delete[] threads;
				  delete[] params;
			  }
		  }
#endif
	  }
	  delete[] changed;
	  delete[] changed2dnum;
	  delete[] inactive;
	  delete[] inactive2dnum;
  }

  (*maxdiff)=0;
  for (i=0;i<totdoc;i++) {
    shrink_state->inactive_since[i]=shrink_state->deactnum-1;
    if(!inconsistent[i]) {
      dist=(lin[i]-model->b)*(float64_t)label[i];
      target=-(learn_parm->eps-(float64_t)label[i]*c[i]);
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
		  delete[] shrink_state->a_history[t];
		  shrink_state->a_history[t]=0;
	  }
  }
}

#endif //USE_SVMLIGHT

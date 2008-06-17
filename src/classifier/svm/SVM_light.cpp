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
/*   Multiple Kernel Learning: Gunnar Raetsch, Soeren Sonnenburg       */
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
	DREAL* lin;
	DREAL* last_lin;
	INT* active;
	INT* docs;
	INT start;
	INT end;
};

struct S_THREAD_PARAM_REACTIVATE_VANILLA
{
	CKernel* kernel;
	DREAL* lin;
	DREAL* aicache;
	DREAL* a;
	DREAL* a_old;
	INT* changed2dnum;
	INT* inactive2dnum;
	INT* label;
	INT start;
	INT end;
};

struct S_THREAD_PARAM 
{
	DREAL * lin ;
	DREAL* W;
	INT start, end;
	INT * active2dnum ;
	INT * docs ;
	CKernel* kernel ;
}  ;

struct S_THREAD_PARAM_KERNEL 
{
	DREAL *Kval ;
	INT *KI, *KJ ;
	INT start, end;
    CKernel * kernel;
}  ;

void* CSVMLight::update_linear_component_linadd_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;
	
	INT jj=0, j=0 ;

	for(jj=params->start;(jj<params->end) && (j=params->active2dnum[jj])>=0;jj++) 
		params->lin[j]+=params->kernel->compute_optimized(params->docs[j]);

	return NULL ;
}

void* CSVMLight::compute_kernel_helper(void* p)
{
	S_THREAD_PARAM_KERNEL* params = (S_THREAD_PARAM_KERNEL*) p;
	
	INT jj=0 ;
	for(jj=params->start;jj<params->end;jj++) 
		params->Kval[jj]=params->kernel->kernel(params->KI[jj], params->KJ[jj]) ;

	return NULL ;
}

void* CSVMLight::update_linear_component_mkl_linadd_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;

	INT num_kernels=params->kernel->get_num_subkernels();

	// determine contributions of different kernels
	for (int i=params->start; i<params->end; i++)
		params->kernel->compute_by_subkernel(i,&(params->W[i*num_kernels]));

	return NULL ;
}

CSVMLight::CSVMLight()
: CSVM()
{
	init();
	set_kernel(NULL);
}

CSVMLight::CSVMLight(DREAL C, CKernel* k, CLabels* lab)
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

#ifdef USE_CPLEX
bool CSVMLight::init_cplex()
{
	while (env==NULL)
	{
		SG_INFO( "trying to initialize CPLEX\n") ;

		int status = 0;
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
					lp = CPXcreateprob (env, &status, "light");

					if ( lp == NULL )
						SG_ERROR( "Failed to create LP.\n");
					else
						CPXchgobjsen (env, lp, CPX_MIN);  /* Problem is minimization */
				}
			}
		}
	}

	return (lp != NULL) && (env != NULL);
}

bool CSVMLight::cleanup_cplex()
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
#endif

CSVMLight::~CSVMLight()
{

  delete[] model->supvec;
  delete[] model->alpha;
  delete[] model->index;
  delete[] model;
  delete[] learn_parm;

  // MKL stuff
  delete[] W ;
  delete[] buffer_num ;
  delete[] buffer_numcols ;

  if (precomputed_subkernels != NULL)
  {
	  for (INT i=0; i<num_precomputed_subkernels; i++)
		  delete[] precomputed_subkernels[i] ;
	  delete[] precomputed_subkernels ;
	  precomputed_subkernels=NULL ;
	  num_precomputed_subkernels=0 ;
  }
  
}

bool CSVMLight::setup_auc_maximization()
{
	SG_INFO( "setting up AUC maximization\n") ;
	
	// get the original labels
	INT num=0;
	ASSERT(labels);
	INT* int_labels=labels->get_int_labels(num);
	ASSERT(kernel->get_num_vec_rhs()==num);
	
	// count positive and negative
	INT num_pos = 0 ;
	INT num_neg = 0 ;
	for (INT i=0; i<num; i++)
		if (int_labels[i]==1)
			num_pos++ ;
		else 
			num_neg++ ;
	
	// create AUC features and labels (alternate labels)
	INT num_auc = num_pos*num_neg ;
	SG_INFO( "num_pos: %i  num_neg: %i  num_auc: %i\n", num_pos, num_neg, num_auc) ;

	WORD* features_auc = new WORD[num_auc*2] ;
	INT* labels_auc = new INT[num_auc] ;
	INT n=0 ;
	for (INT i=0; i<num; i++)
		if (int_labels[i]==1)
			for (INT j=0; j<num; j++)
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
	CWordFeatures* f = new CWordFeatures((INT)0,0) ;
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

	// MKL stuff
	buffer_num = new DREAL[kernel->get_num_vec_rhs()];
	delete[] buffer_numcols ;
	buffer_numcols = NULL ;

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
	SG_DEBUG( "kernel->has_property(KP_LINADD) = %i\n", kernel->has_property(KP_LINADD)) ;
	SG_DEBUG( "kernel->has_property(KP_KERNCOMBINATION) = %i\n", kernel->has_property(KP_KERNCOMBINATION)) ;
	SG_DEBUG( "kernel->has_property(KP_BATCHEVALUATION) = %i\n", kernel->has_property(KP_BATCHEVALUATION)) ;
	SG_DEBUG( "kernel->get_optimization_type() = %s\n", kernel->get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" : "SLOWBUTMEMEFFICIENT" ) ;
	SG_DEBUG( "get_mkl_enabled() = %i\n", get_mkl_enabled()) ;
	SG_DEBUG( "get_linadd_enabled() = %i\n", get_linadd_enabled()) ;
	SG_DEBUG( "get_batch_computation_enabled() = %i\n", get_batch_computation_enabled()) ;
	SG_DEBUG( "kernel->get_num_subkernels() = %i\n", kernel->get_num_subkernels()) ;
	SG_DEBUG( "estimated time: %1.1f minutes\n", 5e-11*pow(kernel->get_num_subkernels(),2.22)*pow(kernel->get_num_vec_rhs(),1.68)*pow(CMath::log2(1/weight_epsilon),2.52)/60) ;

	use_kernel_cache = !(use_precomputed_subkernels || (kernel->get_kernel_type() == K_CUSTOM) ||
						 (get_linadd_enabled() && kernel->has_property(KP_LINADD)) ||
						 kernel->get_precompute_matrix() || 
						 kernel->get_precompute_subkernel_matrix()) ;

	SG_DEBUG( "kernel->get_precompute_matrix() = %i\n", kernel->get_precompute_matrix()) ;
	SG_DEBUG( "kernel->get_precompute_subkernel_matrix() = %i\n", kernel->get_precompute_subkernel_matrix()) ;
	SG_DEBUG( "use_kernel_cache = %i\n", use_kernel_cache) ;

#ifdef USE_CPLEX
	cleanup_cplex();

	if (get_mkl_enabled())
		init_cplex();
#else
	if (get_mkl_enabled())
      SG_ERROR( "CPLEX was disabled at compile-time\n");
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
		INT num = kernel->get_num_vec_rhs();
		INT num_kernels = kernel->get_num_subkernels() ;
		num_precomputed_subkernels=num_kernels ;
		precomputed_subkernels=new SHORTREAL*[num_precomputed_subkernels] ;
		INT num_weights = -1;
		const DREAL* w   = kernel->get_subkernel_weights(num_weights);

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
			precomputed_subkernels[n]=new SHORTREAL[num*(num+1)/2];

		for (INT n=0; n<num_precomputed_subkernels; n++)
		{
			w1[n]=1.0 ;
			kernel->set_subkernel_weights(w1, num_weights) ;

			SHORTREAL * matrix = precomputed_subkernels[n] ;
			
			SG_INFO( "precomputing kernel matrix %i (%ix%i)\n", n, num, num) ;
			for (INT i=0; i<num; i++)
			{
				SG_PROGRESS(i*i,0,num*num);
				
				for (INT j=0; j<=i; j++)
					matrix[i*(i+1)/2+j] = kernel->kernel(i,j) ;

			}
			SG_PROGRESS(num*num,0,num*num);
			SG_DONE();
			w1[n]=0.0 ;
		}

		// restore old weights
		kernel->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;

	}

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

INT CSVMLight::get_runtime() 
{
  clock_t start;
  start = clock();
  return((INT)((double)start*100.0/(double)CLOCKS_PER_SEC));
}

void CSVMLight::svm_learn()
{
	INT *inconsistent, i;
	INT inconsistentnum;
	INT misclassified,upsupvecnum;
	double maxdiff, *lin, *c, *a;
	INT runtime_start,runtime_end;
	INT iterations;
	INT trainpos=0, trainneg=0 ;
	INT totdoc=0;
	ASSERT(labels);
	INT* label=labels->get_int_labels(totdoc);
	ASSERT(label);
	INT* docs=new INT[totdoc];
	delete[] W;
	W=NULL;
	rho=0 ;
	w_gap = 1 ;
	count = 0 ;
	num_rows=0 ;

	if (kernel->has_property(KP_KERNCOMBINATION))
	{
		W = new DREAL[totdoc*kernel->get_num_subkernels()];
		for (i=0; i<totdoc*kernel->get_num_subkernels(); i++)
			W[i]=0;
	}

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

	init_shrink_state(&shrink_state,totdoc,(INT)MAXSHRINK);

	inconsistent = new INT[totdoc];
	c = new double[totdoc];
	a = new double[totdoc];
	a_fullset = new double[totdoc];
	xi_fullset = new double[totdoc];
	lin = new double[totdoc];
	learn_parm->svm_cost = new double[totdoc];

	delete[] model->supvec;
	delete[] model->alpha;
	delete[] model->index;
	model->supvec = new INT[totdoc+2];
	model->alpha = new double[totdoc+2];
	model->index = new INT[totdoc+2];

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
	SG_DEBUG( "alpha:%d num_sv:%d\n", svm_model.alpha, get_num_support_vectors());
  if(svm_model.alpha && get_num_support_vectors()) {
    if(verbosity>=1) {
		SG_INFO( "Computing starting state...");
    }

	DREAL* alpha = new DREAL[totdoc];

	for (i=0; i<totdoc; i++)
		alpha[i]=0;

	for (i=0; i<get_num_support_vectors(); i++)
		alpha[get_support_vector(i)]=get_alpha(i);
	
    INT* index = new INT[totdoc];
    INT* index2dnum = new INT[totdoc+11];
    DREAL* aicache = new DREAL[totdoc];
    for(i=0;i<totdoc;i++) {    /* create full index and clip alphas */
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
				for(i=0;i<totdoc;i++)     /* fill kernel cache with unbounded SV */
					if((alpha[i]>0) && (alpha[i]<learn_parm->svm_cost[i]) 
							&& (kn->kernel_cache_space_available())) 
						kn->cache_kernel_row(i);

				for(i=0;i<totdoc;i++)     /* fill rest of kernel cache with bounded SV */
					if((alpha[i]==learn_parm->svm_cost[i]) 
							&& (kn->kernel_cache_space_available())) 
						kn->cache_kernel_row(i);

				kn = k->get_next_kernel(kn) ;
			}
		}
		else
		{
			for(i=0;i<totdoc;i++)     /* fill kernel cache with unbounded SV */
				if((alpha[i]>0) && (alpha[i]<learn_parm->svm_cost[i]) 
						&& (kernel->kernel_cache_space_available())) 
					kernel->cache_kernel_row(i);

			for(i=0;i<totdoc;i++)     /* fill rest of kernel cache with bounded SV */
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
    for(i=0;i<totdoc;i++) {    /* copy initial alphas */
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
                     &maxdiff,(INT)-1,
                     (INT)1);


	if(verbosity>=1) {
		if(verbosity==1)
		{
			SG_DONE();
			SG_DEBUG("(%ld iterations)", iterations);
		}

		misclassified=0;
		for(i=0;(i<totdoc);i++) { /* get final statistic */
			if((lin[i]-model->b)*(double)label[i] <= 0.0) 
				misclassified++;
		}

		SG_INFO( "Optimization finished (%ld misclassified, maxdiff=%.8f).\n",
				misclassified,maxdiff); 

		SG_INFO( "obj = %.16f, rho = %.16f\n",get_objective(),model->b);
		if (maxdiff>epsilon)
			SG_WARNING( "maximum violation (%f) exceeds svm_epsilon (%f) due to numerical difficulties\n", maxdiff, epsilon); 

		runtime_end=get_runtime();
		upsupvecnum=0;
		for(i=1;i<model->sv_num;i++)
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

INT CSVMLight::optimize_to_convergence(INT* docs, INT* label, INT totdoc, 
			     SHRINK_STATE *shrink_state,
			     INT *inconsistent,
			     double *a, double *lin, double *c, 
			     TIMING *timing_profile, double *maxdiff, 
			     INT heldout, INT retrain)
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
  INT *chosen,*key,i,j,jj,*last_suboptimal_at,noshrink;
  INT inconsistentnum,choosenum,already_chosen=0,iteration;
  INT misclassified,supvecnum=0,*active2dnum,inactivenum;
  INT *working2dnum,*selexam;
  INT activenum;
  double criterion, eq;
  double *a_old;
  INT t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  INT transductcycle;
  INT transduction;
  double epsilon_crit_org; 
  double bestmaxdiff;
  double worstmaxdiff;
  INT   bestmaxdiffiter,terminate;
  bool reactivated=false;

  double *selcrit;  /* buffer for sorting */        
  DREAL *aicache;  /* buffer to keep one row of hessian */
  QP qp;            /* buffer for one quadratic program */

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
  if(kernel->has_property(KP_LINADD) && get_linadd_enabled()) {
	  learn_parm->epsilon_crit=2.0;
      /* caching makes no sense for linear kernel */
  } 
  learn_parm->epsilon_shrink=2;
  (*maxdiff)=1;

  SG_DEBUG("totdoc:%d\n",totdoc);
  chosen = new INT[totdoc];
  last_suboptimal_at =new INT[totdoc];
  key =new INT[totdoc+11];
  selcrit =new double[totdoc];
  selexam =new INT[totdoc];
  a_old =new double[totdoc];
  aicache =new DREAL[totdoc];
  working2dnum =new INT[totdoc+11];
  active2dnum =new INT[totdoc+11];
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
  
  kernel->set_time(iteration);  /* for lru cache */

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
  CTime start_time; 

#ifdef CYGWIN
  for(;((iteration<3) || (retrain && (!terminate))||((w_gap>get_weight_epsilon()) && get_mkl_enabled())); iteration++){
#else
  CSignal::clear() ;
  for(;((!CSignal::cancel_computations()) && ((iteration<3) || (retrain && (!terminate))||((w_gap>get_weight_epsilon()) && get_mkl_enabled()))); iteration++){
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
			  if(CMath::min(learn_parm->svm_newvarsinqp, learn_parm->svm_maxqpsize-choosenum)>=4 &&
					  (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())))
			  {
				  /* select part of the working set from cache */
				  already_chosen=select_next_qp_subproblem_grad(
					  label,a,lin,c,totdoc,
					  (INT)(CMath::min(learn_parm->svm_maxqpsize-choosenum,
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

	  for(jj=0;(i=working2dnum[jj])>=0;jj++) {
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
		  /* INT time no progress? */
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
				  CMath::max((INT)(activenum/10),
					  CMath::max((INT)(totdoc/500),(INT) 100)),
				  a,inconsistent, c, lin, label);

		  inactivenum=totdoc-activenum;

		  if (use_kernel_cache && (supvecnum>kernel->get_max_elems_cache()) 
				  && ((kernel->get_activenum_cache()-activenum)>CMath::max((INT)(activenum/10),(INT) 500))) {

			  kernel->kernel_cache_shrink(totdoc, CMath::min((INT) (kernel->get_activenum_cache()-activenum),
						  (INT) (kernel->get_activenum_cache()-supvecnum)),
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

double CSVMLight::compute_objective_function(double *a, double *lin, double *c, double eps, INT *label, INT totdoc)
     /* Return value of objective function. */
     /* Works only relative to the active variables! */
{
  /* calculate value of objective function */
  double criterion=0;

  for(INT i=0;i<totdoc;i++)
	  criterion=criterion+(eps-(double)label[i]*c[i])*a[i]+0.5*a[i]*label[i]*lin[i];


  /*double check=0;
  for(INT i=0;i<totdoc;i++)
  {
	  check+=a[i]*eps-a[i]*label[i];
	  for(INT j=0;j<totdoc;j++)
		  check+= 0.5*a[i]*label[i]*a[j]*label[j]*kernel->kernel(i,j);
  }

  SG_INFO("CLASSIFICATION OBJECTIVE %f vs. CHECK %f (diff %f)\n", criterion, check, criterion-check);
  */

  return(criterion);
}


void CSVMLight::clear_index(INT *index)  
              /* initializes and empties index */
{
  index[0]=-1;
} 

void CSVMLight::add_to_index(INT *index, INT elem)
     /* initializes and empties index */
{
  register INT i;
  for(i=0;index[i] != -1;i++);
  index[i]=elem;
  index[i+1]=-1;
}

INT CSVMLight::compute_index(INT *binfeature, INT range, INT *index)
     /* create an inverted index of binfeature */
{               
  register INT i,ii;

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


void CSVMLight::optimize_svm(INT* docs, INT* label,
		  INT *exclude_from_eq_const, double eq_target,
		  INT *chosen, INT *active2dnum,
		  INT totdoc, INT *working2dnum, INT varnum, 
		  double *a, double *lin, double *c,
		  DREAL *aicache, QP *qp, 
		  double *epsilon_crit_target)
     /* Do optimization on the working set. */
{
    INT i;
    double *a_v;

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

    for(i=0;i<varnum;i++)
      a[working2dnum[i]]=a_v[i];
}

void CSVMLight::compute_matrices_for_optimization_parallel(INT* docs, INT* label, 
														   INT *exclude_from_eq_const, double eq_target,
														   INT *chosen, INT *active2dnum, 
														   INT *key, double *a, double *lin, double *c, 
														   INT varnum, INT totdoc,
														   DREAL *aicache, QP *qp)
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
		register INT ki,kj,i,j;
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

		ASSERT(parallel.get_num_threads()>1);
		INT *KI=new INT[varnum*varnum] ;
		INT *KJ=new INT[varnum*varnum] ;
		INT Knum=0 ;
		DREAL *Kval = new DREAL[varnum*(varnum+1)/2] ;
		for(i=0;i<varnum;i++) {
			ki=key[i];
			KI[Knum]=docs[ki] ;
			KJ[Knum]=docs[ki] ;
			Knum++ ;
			for(j=i+1;j<varnum;j++) 
			{
				kj=key[j];
				KI[Knum]=docs[ki] ;
				KJ[Knum]=docs[kj] ;
				Knum++ ;
			}
		}
		ASSERT(Knum<=varnum*(varnum+1)/2);

		pthread_t threads[parallel.get_num_threads()-1];
		S_THREAD_PARAM_KERNEL params[parallel.get_num_threads()-1];
		INT step= Knum/parallel.get_num_threads();
		//SG_DEBUG( "\nkernel-step size: %i\n", step) ;
		for (INT t=0; t<parallel.get_num_threads()-1; t++)
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

		for (INT t=0; t<parallel.get_num_threads()-1; t++)
			pthread_join(threads[t], NULL);

		Knum=0 ;
		for(i=0;i<varnum;i++) {
			ki=key[i];

			/* Compute the matrix for equality constraints */
			qp->opt_ce[i]=label[ki];
			qp->opt_low[i]=0;
			qp->opt_up[i]=learn_parm->svm_cost[ki];

			kernel_temp=Kval[Knum] ;
			Knum++ ;
			/* compute linear part of objective function */
			qp->opt_g0[i]-=(kernel_temp*a[ki]*(double)label[ki]); 
			/* compute quadratic part of objective function */
			qp->opt_g[varnum*i+i]=kernel_temp;

			for(j=i+1;j<varnum;j++) {
				kj=key[j];
				kernel_temp=Kval[Knum] ;
				Knum++ ;
				/* compute linear part of objective function */
				qp->opt_g0[i]-=(kernel_temp*a[kj]*(double)label[kj]);
				qp->opt_g0[j]-=(kernel_temp*a[ki]*(double)label[ki]); 
				/* compute quadratic part of objective function */
				qp->opt_g[varnum*i+j]=(double)label[ki]*(double)label[kj]*kernel_temp;
				qp->opt_g[varnum*j+i]=qp->opt_g[varnum*i+j];//(double)label[ki]*(double)label[kj]*kernel_temp;
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

		for(i=0;i<varnum;i++) {
			/* assure starting at feasible point */
			qp->opt_xinit[i]=a[key[i]];
			/* set linear part of objective function */
			qp->opt_g0[i]=(learn_parm->eps-(double)label[key[i]]*c[key[i]])+qp->opt_g0[i]*(double)label[key[i]];    
		}

		if(verbosity>=3) {
			SG_DONE();
		}
	}
#endif
}

void CSVMLight::compute_matrices_for_optimization(INT* docs, INT* label, 
												  INT *exclude_from_eq_const, double eq_target,
												  INT *chosen, INT *active2dnum, 
												  INT *key, double *a, double *lin, double *c, 
												  INT varnum, INT totdoc,
												  DREAL *aicache, QP *qp)
{
  register INT ki,kj,i,j;
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
	  
	  kernel_temp=compute_kernel(docs[ki], docs[ki]); 
	  /* compute linear part of objective function */
	  qp->opt_g0[i]-=(kernel_temp*a[ki]*(double)label[ki]); 
	  /* compute quadratic part of objective function */
	  qp->opt_g[varnum*i+i]=kernel_temp;
	  
	  for(j=i+1;j<varnum;j++) {
		  kj=key[j];
		  kernel_temp=compute_kernel(docs[ki], docs[kj]);

		  /* compute linear part of objective function */
		  qp->opt_g0[i]-=(kernel_temp*a[kj]*(double)label[kj]);
		  qp->opt_g0[j]-=(kernel_temp*a[ki]*(double)label[ki]); 
		  /* compute quadratic part of objective function */
		  qp->opt_g[varnum*i+j]=(double)label[ki]*(double)label[kj]*kernel_temp;
		  qp->opt_g[varnum*j+i]=qp->opt_g[varnum*i+j];//(double)label[ki]*(double)label[kj]*kernel_temp;
	  }
	  
	  if(verbosity>=3) {
		  if(i % 20 == 0) {
			  SG_DEBUG( "%ld..",i);
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
	  SG_DONE();
  }
}


INT CSVMLight::calculate_svm_model(INT* docs, INT *label,
			 double *lin, double *a, double *a_old, double *c, 
			 INT *working2dnum, INT *active2dnum)
     /* Compute decision function based on current values */
     /* of alpha. */
{
  INT i,ii,pos,b_calculated=0,first_low,first_high;
  double ex_c,b_temp,b_low,b_high;

  if(verbosity>=3) {
   SG_DEBUG( "Calculating model...");
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
   SG_DONE();
  }

  return(model->sv_num-1); /* have to substract one, since element 0 is empty*/
}

INT CSVMLight::check_optimality(INT* label,
		      double *a, double *lin, double *c, INT totdoc, 
		      double *maxdiff, double epsilon_crit_org, INT *misclassified, 
		      INT *inconsistent, INT *active2dnum,
		      INT *last_suboptimal_at, 
		      INT iteration)
     /* Check KT-conditions */
{
  INT i,ii,retrain;
  double dist,ex_c,target;

  if (kernel->has_property(KP_LINADD) && get_linadd_enabled())
	  learn_parm->epsilon_shrink=-learn_parm->epsilon_crit+epsilon_crit_org;
  else
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
		  /* Count how INT a variable was at lower/upper bound (and optimal).*/
		  /* Variables, which were at the bound and optimal for a INT */
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

void CSVMLight::update_linear_component_mkl(INT* docs, INT* label, 
											INT *active2dnum, double *a, 
											double *a_old, INT *working2dnum, 
											INT totdoc,
											double *lin, DREAL *aicache)
{
	INT num = kernel->get_num_vec_rhs();
	INT num_weights = -1;
	INT num_kernels = kernel->get_num_subkernels() ;
	const DREAL* w   = kernel->get_subkernel_weights(num_weights);

	ASSERT(num_weights==num_kernels);
	DREAL* sumw=new DREAL[num_kernels];
	DREAL suma=-17;

	if (use_precomputed_subkernels) // everything is already precomputed
	{
		ASSERT(precomputed_subkernels);
		for (INT n=0; n<num_kernels; n++)
		{
			ASSERT(precomputed_subkernels[n]);
			SHORTREAL * matrix = precomputed_subkernels[n] ;
			for(INT i=0;i<num;i++) 
			{
				if(a[i] != a_old[i]) 
				{
					for(INT j=0;j<num;j++) 
						if (i>=j)
							W[j*num_kernels+n]+=(a[i]-a_old[i])*matrix[i*(i+1)/2+j]*(double)label[i];
						else
							W[j*num_kernels+n]+=(a[i]-a_old[i])*matrix[i+j*(j+1)/2]*(double)label[i];
				}
			}
		}
	} 
	else if ((kernel->get_kernel_type()==K_COMBINED) && 
			 (!((CCombinedKernel*)kernel)->get_append_subkernel_weights()))// for combined kernel
	{
		CCombinedKernel* k      = (CCombinedKernel*) kernel;
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
						W[j*num_kernels+n]+=(a[i]-a_old[i])*aicache[j]*(double)label[i];
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
			kernel->set_subkernel_weights(w1, num_weights) ;
		
			for(INT i=0;i<num;i++) 
			{
				if(a[i] != a_old[i]) 
				{
					for(INT j=0;j<num;j++) 
						W[j*num_kernels+n]+=(a[i]-a_old[i])*kernel->kernel(i,j)*(double)label[i];
				}
			}
			w1[n]=0.0 ;
		}

		// restore old weights
		kernel->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;
	}
	
	DREAL mkl_objective=0;
#ifdef HAVE_LAPACK
	DREAL *alphay  = buffer_num ;
	suma = 0 ;
	
	for (int i=0; i<num; i++)
	{
		alphay[i]=a[i]*label[i] ;
		suma+=a[i] ;
	}

	for (int i=0; i<num_kernels; i++)
		sumw[i]=0;
	
	cblas_dgemv(CblasColMajor, CblasNoTrans, num_kernels, num,
				0.5, W, num_kernels, alphay, 1, 1.0, sumw, 1) ;
	
	mkl_objective=-suma;
	for (int i=0; i<num_kernels; i++)
		mkl_objective+=w[i]*sumw[i] ;
#else
	suma=0;
	for(int i=0; i<num; i++)
		suma += a[i];

	mkl_objective=-suma;
	for (int d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for(int i=0; i<num; i++)
			sumw[d] += a[i]*(0.5*label[i]*W[i*num_kernels+d]);
		mkl_objective   += w[d]*(sumw[d]);
	}
#endif
	
	count++ ;
#ifdef USE_CPLEX			
	w_gap = CMath::abs(1-rho/mkl_objective) ;
	
	if ((w_gap >= 0.9999*get_weight_epsilon()))
	{
		if (!lp_initialized)
		{
			SG_INFO( "creating LP\n") ;
			
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
            SG_ERROR( "%s", errmsg);
			}
			
			// add constraint sum(w)=1 ;
			SG_INFO( "adding the first row\n") ;
			int initial_rmatbeg[1] ;
			int initial_rmatind[num_kernels+1] ;
			double initial_rmatval[num_kernels+1] ;
			double initial_rhs[1] ;
			char initial_sense[1] ;
			
			 // 1-norm MKL
			if (mkl_norm==1)
			{
				initial_rmatbeg[0] = 0;
				initial_rhs[0]=1 ;     // rhs=1 ;
				initial_sense[0]='E' ; // equality

				for (INT i=0; i<num_kernels; i++)
				{
					initial_rmatind[i]=i ;
					initial_rmatval[i]=1 ;
				}
				initial_rmatind[num_kernels]=2*num_kernels ;
				initial_rmatval[num_kernels]=0 ;

				status = CPXaddrows (env, lp, 0, 1, num_kernels+1, 
						initial_rhs, initial_sense, initial_rmatbeg,
						initial_rmatind, initial_rmatval, NULL, NULL);

			}
			else // 2-norm MKL
			{
				initial_rmatbeg[0] = 0;
				initial_rhs[0]=0 ;     // rhs=1 ;
				initial_sense[0]='L' ; // equality

				initial_rmatind[0]=2*num_kernels ;
				initial_rmatval[0]=0 ;

				status = CPXaddrows (env, lp, 0, 1, 1, 
						initial_rhs, initial_sense, initial_rmatbeg,
						initial_rmatind, initial_rmatval, NULL, NULL);

				for (INT i=0; i<num_kernels; i++)
				{
					initial_rmatind[i]=i ;
					initial_rmatval[i]=1 ;
				}
				initial_rmatind[num_kernels]=2*num_kernels ;
				initial_rmatval[num_kernels]=0 ;

				status = CPXaddqconstr (env, lp, 0, num_kernels+1, 1.0, 'L', NULL, NULL,
						initial_rmatind, initial_rmatind, initial_rmatval, NULL);
			}

			if ( status )
				SG_ERROR( "Failed to add the first row.\n");

			lp_initialized = true ;
			
			if (C_mkl!=0.0)
			{
				for (INT q=0; q<num_kernels-1; q++)
				{
					int rmatbeg[1] ;
					int rmatind[3] ;
					double rmatval[3] ;
					double rhs[1] ;
					char sense[1] ;
					
					rmatbeg[0] = 0;
					rhs[0]=0 ;     // rhs=1 ;
					sense[0]='L' ; // <=
					rmatind[0]=q ;
					rmatval[0]=1 ;
					rmatind[1]=q+1 ;
					rmatval[1]=-1 ;
					rmatind[2]=num_kernels+q ;
					rmatval[2]=-1 ;
					status = CPXaddrows (env, lp, 0, 1, 3, 
										 rhs, sense, rmatbeg,
										 rmatind, rmatval, NULL, NULL);
					if ( status )
						SG_ERROR( "Failed to add a smothness row (1).\n");
					
					rmatbeg[0] = 0;
					rhs[0]=0 ;     // rhs=1 ;
					sense[0]='L' ; // <=
					rmatind[0]=q ;
					rmatval[0]=-1 ;
					rmatind[1]=q+1 ;
					rmatval[1]=1 ;
					rmatind[2]=num_kernels+q ;
					rmatval[2]=-1 ;
					status = CPXaddrows (env, lp, 0, 1, 3, 
										 rhs, sense, rmatbeg,
										 rmatind, rmatval, NULL, NULL);
					if ( status )
						SG_ERROR( "Failed to add a smothness row (2).\n");
				}
			}
		}

		SG_DEBUG( "*") ;
		
		{ // add the new row
			//SG_INFO( "add the new row\n") ;
			
			int rmatbeg[1] ;
			int rmatind[num_kernels+1] ;
			double rmatval[num_kernels+1] ;
			double rhs[1] ;
			char sense[1] ;
			
			rmatbeg[0] = 0;
			if (mkl_norm==1)
				rhs[0]=0 ;
			else
				rhs[0]=-suma ;
			sense[0]='L' ;
			
			for (INT i=0; i<num_kernels; i++)
			{
				rmatind[i]=i ;
				if (mkl_norm==1)
					rmatval[i]=-(sumw[i]-suma) ;
				else
					rmatval[i]=-sumw[i];
			}
			rmatind[num_kernels]=2*num_kernels ;
			rmatval[num_kernels]=-1 ;
			
			INT status = CPXaddrows (env, lp, 0, 1, num_kernels+1, 
									 rhs, sense, rmatbeg,
									 rmatind, rmatval, NULL, NULL);
			if ( status ) 
				SG_ERROR( "Failed to add the new row.\n");
		}
		
		{ 
			
			INT status;

			if (mkl_norm==1) // optimize 1 norm MKL
				status = CPXlpopt (env, lp);
			else // optimize 2 norm MKL
				status = CPXbaropt(env, lp);

			if ( status ) 
				SG_ERROR( "Failed to optimize Problem.\n");
			
			// obtain solution
			INT cur_numrows = CPXgetnumrows (env, lp);
			INT cur_numcols = CPXgetnumcols (env, lp);
			num_rows = cur_numrows ;
			
			if (!buffer_numcols)
				buffer_numcols  = new DREAL[cur_numcols] ;
					
			DREAL *x     = buffer_numcols ;
			DREAL *slack = new DREAL[cur_numrows] ;
			DREAL *pi    = NULL;
			if (use_mkl==1)
				pi=new DREAL[cur_numrows];
			
			if ( x     == NULL ||
				 slack == NULL ||
				 pi    == NULL   ) {
				status = CPXERR_NO_MEMORY;
            SG_ERROR( "Could not allocate memory for solution.\n");
			}
			INT solstat = 0 ;
			DREAL objval = 0 ;

			if (mkl_norm==1)
				status = CPXsolution (env, lp, &solstat, &objval, x, pi, slack, NULL);
			else
				status = CPXsolution (env, lp, &solstat, &objval, x, NULL, slack, NULL);
			INT solution_ok = (!status) ;
			if ( status )
				SG_ERROR( "Failed to obtain solution.\n");
			
			num_active_rows=0 ;
			if (solution_ok)
			{
				/* 1 norm mkl */
				DREAL max_slack = -CMath::INFTY ;
				INT max_idx = -1 ;
				INT start_row = 1 ;
				if (C_mkl!=0.0)
					start_row+=2*(num_kernels-1);

				for (INT i = start_row; i < cur_numrows; i++)  // skip first
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
					else if (mkl_norm==2)
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
					num_active_rows++ ;

				}
				
				// have at most max(100,num_active_rows*2) rows, if not, remove one
				if ( (num_rows-start_row>CMath::max(100,2*num_active_rows)) && (max_idx!=-1))
				{
					//SG_INFO( "-%i(%i,%i)",max_idx,start_row,num_rows) ;
					status = CPXdelrows (env, lp, max_idx, max_idx) ;
					if ( status ) 
                  SG_ERROR( "Failed to remove an old row.\n");
				}

				// have at most max(100,num_active_rows*2) rows, if not, remove one
				if ( (num_rows-start_row>CMath::max(100,2*num_active_rows)) && (max_idx!=-1))
				{
					SG_DEBUG("Removing row %i (slack=%f, start_row=%i, num_rows=%i)",
							max_idx, max_slack, start_row,num_rows) ;
					status = CPXdelrows (env, lp, max_idx, max_idx) ;
					if ( status ) 
						SG_ERROR( "Failed to remove an old row.\n");
				}


				// set weights, store new rho and compute new w gap
				kernel->set_subkernel_weights(x, num_kernels) ;
				rho = -x[2*num_kernels] ;
				w_gap = CMath::abs(1-rho/mkl_objective) ;
				
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
		SG_DEBUG("\n%i. OBJ: %f  RHO: %f  wgap=%f agap=%f (activeset=%i; active rows=%i/%i)\n", count, mkl_objective,rho,w_gap,mymaxdiff,jj,num_active_rows,num_rows-start_row);
	}
	
	delete[] sumw;
}


void CSVMLight::update_linear_component_mkl_linadd(INT* docs, INT* label, 
												   INT *active2dnum, double *a, 
												   double *a_old, INT *working2dnum, 
												   INT totdoc,
												   double *lin, DREAL *aicache)
{
	// kernel with LP_LINADD property is assumed to have 
	// compute_by_subkernel functions
	INT num = kernel->get_num_vec_rhs();
	INT num_weights = -1;
	INT num_kernels = kernel->get_num_subkernels() ;
	const DREAL* w   = kernel->get_subkernel_weights(num_weights);
	
	ASSERT(num_weights==num_kernels);
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
		kernel->set_subkernel_weights(w1, num_weights) ;
		
		// create normal update (with changed alphas only)
		kernel->clear_normal();
		for(INT ii=0, i=0;(i=working2dnum[ii])>=0;ii++) {
			if(a[i] != a_old[i]) {
				kernel->add_to_normal(docs[i], (a[i]-a_old[i])*(double)label[i]);
			}
		}

		if (parallel.get_num_threads() < 2)
		{
			// determine contributions of different kernels
			for (int i=0; i<num; i++)
				kernel->compute_by_subkernel(i,&W[i*num_kernels]);
		}
#ifndef WIN32
		else
		{
			pthread_t threads[parallel.get_num_threads()-1];
			S_THREAD_PARAM params[parallel.get_num_threads()-1];
			INT step= num/parallel.get_num_threads();

			for (INT t=0; t<parallel.get_num_threads()-1; t++)
			{
				params[t].kernel = kernel;
				params[t].W = W;
				params[t].start = t*step;
				params[t].end = (t+1)*step;
				pthread_create(&threads[t], NULL, CSVMLight::update_linear_component_mkl_linadd_helper, (void*)&params[t]);
			}

			for (int i=params[parallel.get_num_threads()-2].end; i<num; i++)
				kernel->compute_by_subkernel(i,&W[i*num_kernels]);

			for (INT t=0; t<parallel.get_num_threads()-1; t++)
				pthread_join(threads[t], NULL);
		}
#endif

		// restore old weights
		kernel->set_subkernel_weights(w_backup,num_weights);
		
		delete[] w_backup;
		delete[] w1;
	}
	DREAL mkl_objective=0;
#ifdef HAVE_LAPACK
	DREAL *alphay  = buffer_num;
	DREAL sumalpha = 0;
	
	for (int i=0; i<num; i++)
	{
		alphay[i]=a[i]*label[i];
		sumalpha-=a[i];
	}
	for (int i=0; i<num_kernels; i++)
		sumw[i]=sumalpha;
	
	cblas_dgemv(CblasColMajor, CblasNoTrans, num_kernels, num,
				0.5, W, num_kernels, alphay, 1, 1.0, sumw, 1);
	
	for (int i=0; i<num_kernels; i++)
		mkl_objective+=w[i]*sumw[i];
#else
	for (int d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for(int i=0; i<num; i++)
			sumw[d] += a[i]*(0.5*label[i]*W[i*num_kernels+d] - 1);
		mkl_objective   += w[d]*sumw[d];
	}
#endif
	
	count++ ;
#ifdef USE_CPLEX			
	w_gap = CMath::abs(1-rho/mkl_objective) ;

	if ((w_gap >= 0.9999*get_weight_epsilon()))// && (mymaxdiff < prev_mymaxdiff/2.0))
	{
		SG_DEBUG( "*") ;
		if (!lp_initialized)
		{
			SG_INFO( "creating LP\n") ;
			
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
            SG_ERROR( "%s", errmsg);
			}
			
			// add constraint sum(w)=1 ;
			int initial_rmatbeg[1] ;
			int initial_rmatind[num_kernels+1] ;
			double initial_rmatval[num_kernels+1] ;
			double initial_rhs[1] ;
			char initial_sense[1] ;
			
			initial_rmatbeg[0] = 0;
			initial_rhs[0]=1 ;     // rhs=1 ;
			initial_sense[0]='E' ; // equality
			
			for (INT i=0; i<num_kernels; i++)
			{
				initial_rmatind[i]=i ;
				initial_rmatval[i]=1 ;
			}
			initial_rmatind[num_kernels]=2*num_kernels ;
			initial_rmatval[num_kernels]=0 ;
			
			status = CPXaddrows (env, lp, 0, 1, num_kernels+1, 
								 initial_rhs, initial_sense, initial_rmatbeg,
								 initial_rmatind, initial_rmatval, NULL, NULL);
			if ( status ) {
            SG_ERROR( "Failed to add the first row.\n");
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
                  SG_ERROR( "Failed to add a smothness row (1).\n");
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
                  SG_ERROR( "Failed to add a smothness row (2).\n");
					}
				}
			}
		}
		
		{ // add the new row
			//SG_INFO( "add the new row\n") ;
			
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
            SG_ERROR( "Failed to add the new row.\n");
		}
		
		{ // optimize
			INT status = CPXlpopt (env, lp);
			if ( status ) 
            SG_ERROR( "Failed to optimize LP.\n");
			
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
            SG_ERROR( "Could not allocate memory for solution.\n");
			}
			INT solstat = 0 ;
			DREAL objval = 0 ;
			status = CPXsolution (env, lp, &solstat, &objval, x, pi, slack, NULL);
			INT solution_ok = (!status) ;
			if ( status ) {
            SG_ERROR( "Failed to obtain solution.\n");
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
					//SG_INFO( "-%i(%i,%i)",max_idx,start_row,num_rows) ;
					status = CPXdelrows (env, lp, max_idx, max_idx) ;
					if ( status ) 
                  SG_ERROR( "Failed to remove an old row.\n");
				}

				// set weights, store new rho and compute new w gap
				kernel->set_subkernel_weights(x, num_kernels) ;
				rho = -x[2*num_kernels] ;
				w_gap = CMath::abs(1-rho/mkl_objective) ;
				
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
		SG_DEBUG("\n%i. OBJ: %f  RHO: %f  wgap=%f agap=%f (activeset=%i; active rows=%i/%i)\n", count, mkl_objective,rho,w_gap,mymaxdiff,jj,num_active_rows,num_rows-start_row);
	}
	
	delete[] sumw;
}

void CSVMLight::update_linear_component(INT* docs, INT* label, 
										INT *active2dnum, double *a, 
										double *a_old, INT *working2dnum, 
										INT totdoc,
										double *lin, DREAL *aicache, double* c)
     /* keep track of the linear component */
     /* lin of the gradient etc. by updating */
     /* based on the change of the variables */
     /* in the current working set */
{
	register INT i=0,ii=0,j=0,jj=0;

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

			INT num_working=0;
			for(ii=0;(i=working2dnum[ii])>=0;ii++) {
				if(a[i] != a_old[i]) {
					kernel->add_to_normal(docs[i], (a[i]-a_old[i])*(double)label[i]);
					num_working++;
				}
			}

			if (num_working>0)
			{
				if (parallel.get_num_threads() < 2)
				{
					for(jj=0;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=kernel->compute_optimized(docs[j]);
					}
				}
#ifndef WIN32
				else
				{
					INT num_elem = 0 ;
					for(jj=0;(j=active2dnum[jj])>=0;jj++) num_elem++ ;

					pthread_t threads[parallel.get_num_threads()-1] ;
					S_THREAD_PARAM params[parallel.get_num_threads()-1] ;
					INT start = 0 ;
					INT step = num_elem/parallel.get_num_threads();
					INT end = step ;

					for (INT t=0; t<parallel.get_num_threads()-1; t++)
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

					for(jj=params[parallel.get_num_threads()-2].end;(j=active2dnum[jj])>=0;jj++) {
						lin[j]+=kernel->compute_optimized(docs[j]);
					}
					void* ret;
					for (INT t=0; t<parallel.get_num_threads()-1; t++)
						pthread_join(threads[t], &ret) ;
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
			for(jj=0;(i=working2dnum[jj])>=0;jj++) {
				if(a[i] != a_old[i]) {
					kernel->get_kernel_row(i,active2dnum,aicache);
					for(ii=0;(j=active2dnum[ii])>=0;ii++)
						lin[j]+=(a[i]-a_old[i])*aicache[j]*(double)label[i];
				}
			}
		}
	}
}


/*************************** Working set selection ***************************/

INT CSVMLight::select_next_qp_subproblem_grad(INT* label, 
											   double *a, double *lin, 
											   double *c, INT totdoc, 
											   INT qp_size, 
											   INT *inconsistent, 
											   INT *active2dnum, 
											   INT *working2dnum, 
											   double *selcrit, 
											   INT *select, 
											   INT cache_only,
											   INT *key, INT *chosen)
	/* Use the feasible direction approach to select the next
	   qp-subproblem (see chapter 'Selecting a good working set'). If
	   'cache_only' is true, then the variables are selected only among
	   those for which the kernel evaluations are cached. */
{
	INT choosenum,i,j,k,activedoc,inum,valid;
	double s;
	
	for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
	choosenum=0;
	activedoc=0;
	for(i=0;(j=active2dnum[i])>=0;i++) {
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
			selcrit[activedoc]=(double)label[j]*(learn_parm->eps-(double)label[j]*c[j]+(double)label[j]*lin[j]);
			key[activedoc]=j;
			activedoc++;
		}
	}
	select_top_n(selcrit,activedoc,select,(INT)(qp_size/2));
	for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
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
	for(i=0;(j=active2dnum[i])>=0;i++) {
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
			selcrit[activedoc]=-(double)label[j]*(learn_parm->eps-(double)label[j]*c[j]+(double)label[j]*lin[j]);
			/*  selcrit[activedoc]=-(double)(label[j]*(-1.0+(double)label[j]*lin[j])); */
			key[activedoc]=j;
			activedoc++;
		}
	}
	select_top_n(selcrit,activedoc,select,(INT)(qp_size/2));
	for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
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

INT CSVMLight::select_next_qp_subproblem_rand(INT* label, 
				    double *a, double *lin, 
				    double *c, INT totdoc, 
				    INT qp_size, 
				    INT *inconsistent, 
				    INT *active2dnum, 
				    INT *working2dnum, 
				    double *selcrit, 
				    INT *select, 
				    INT *key, 
				    INT *chosen, 
				    INT iteration)
/* Use the feasible direction approach to select the next
   qp-subproblem (see section 'Selecting a good working set'). Chooses
   a feasible direction at (pseudo) random to help jump over numerical
   problem. */
{
  INT choosenum,i,j,k,activedoc,inum;
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
  select_top_n(selcrit,activedoc,select,(INT)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
	if (use_kernel_cache)
		kernel->kernel_cache_touch(i); /* make sure it does not get kicked */
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
  select_top_n(selcrit,activedoc,select,(INT)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
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



void CSVMLight::select_top_n(double *selcrit, INT range, INT* select,
		  INT n)
{
  register INT i,j;

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

void CSVMLight::init_shrink_state(SHRINK_STATE *shrink_state, INT totdoc,
		       INT maxhistory)
{
  INT i;

  shrink_state->deactnum=0;
  shrink_state->active = new INT[totdoc];
  shrink_state->inactive_since = new INT[totdoc];
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

INT CSVMLight::shrink_problem(SHRINK_STATE *shrink_state, 
		    INT *active2dnum, 
		    INT *last_suboptimal_at, 
		    INT iteration, 
		    INT totdoc, 
		    INT minshrink, 
		    double *a, 
		    INT *inconsistent,
			double* c,
			double* lin,
			int* label)
     /* Shrink some variables away.  Do the shrinking only if at least
        minshrink variables can be removed. */
{
  INT i,ii,change,activenum,lastiter;
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
		  SG_INFO( " Shrinking...");
	  }

	  if (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())) { /*  non-linear case save alphas */
	 
		  a_old=new double[totdoc];
		  shrink_state->a_history[shrink_state->deactnum]=a_old;
		  for(i=0;i<totdoc;i++) {
			  a_old[i]=a[i];
		  }
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
	DREAL* lin = params->lin;
	DREAL* last_lin = params->last_lin;
	INT* active = params->active;
	INT* docs = params->docs;
	INT start = params->start;
	INT end = params->end;

	for(INT i=start;i<end;i++)
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
	DREAL* lin = params->lin;
	DREAL* aicache = params->aicache;
	DREAL* a= params->a;
	DREAL* a_old = params->a_old;
	INT* changed2dnum = params->changed2dnum;
	INT* inactive2dnum = params->inactive2dnum;
	INT* label = params->label;
	INT start = params->start;
	INT end = params->end;

	for(INT ii=start;ii<end;ii++)
	{
		INT i=changed2dnum[ii];
		INT j=0;
		ASSERT(i>=0);

		k->get_kernel_row(i,inactive2dnum,aicache);
		for(INT jj=0;(j=inactive2dnum[jj])>=0;jj++)
			lin[j]+=(a[i]-a_old[i])*aicache[j]*(double)label[i];
	}
	return NULL;
}

void CSVMLight::reactivate_inactive_examples(INT* label, 
				  double *a, 
				  SHRINK_STATE *shrink_state, 
				  double *lin, 
				  double *c, 
				  INT totdoc, 
				  INT iteration, 
				  INT *inconsistent, 
				  INT* docs, 
				  DREAL *aicache, 
				  double *maxdiff)
     /* Make all variables active again which had been removed by
        shrinking. */
     /* Computes lin for those variables from scratch. */
{
  register INT i,j,ii,jj,t,*changed2dnum,*inactive2dnum;
  INT *changed,*inactive;
  register double *a_old,dist;
  double ex_c,target;

  if (kernel->has_property(KP_LINADD) && get_linadd_enabled()) { /* special linear case */
	  a_old=shrink_state->last_a;    

	  if (!use_batch_computation || !kernel->has_property(KP_BATCHEVALUATION))
	  {
		  SG_DEBUG( " clear normal - linadd\n");
		  kernel->clear_normal();

		  INT num_modified=0;
		  for(i=0;i<totdoc;i++) {
			  if(a[i] != a_old[i]) {
				  kernel->add_to_normal(docs[i], ((a[i]-a_old[i])*(double)label[i]));
				  a_old[i]=a[i];
				  num_modified++;
			  }
		  }
		  
		  if (num_modified>0)
		  {
			  INT num_threads=parallel.get_num_threads();
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
				  pthread_t threads[num_threads-1];
				  S_THREAD_PARAM_REACTIVATE_LINADD params[num_threads];
				  INT step= totdoc/num_threads;

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

			  }
#endif

		  }
	  }
	  else 
	  {
		  DREAL *alphas = new DREAL[totdoc] ;
		  INT *idx = new INT[totdoc] ;
		  INT num_suppvec=0 ;

		  for (i=0; i<totdoc; i++)
		  {
			  if(a[i] != a_old[i]) 
			  {
				  alphas[num_suppvec] = (a[i]-a_old[i])*(double)label[i];
				  idx[num_suppvec] = i ;
				  a_old[i] = a[i] ;
				  num_suppvec++ ;
			  }
		  }

		  if (num_suppvec>0)
		  {
			  INT num_inactive=0;
			  INT* inactive_idx=new INT[totdoc]; // infact we only need a subset 

			  j=0;
			  for(i=0;i<totdoc;i++) 
			  {
				  if(!shrink_state->active[i])
				  {
					  inactive_idx[j++] = i;
					  num_inactive++;
				  }
			  }

			  if (num_inactive>0)
			  {
				  DREAL* dest = new DREAL[num_inactive];
				  memset(dest, 0, sizeof(DREAL)*num_inactive);

				  kernel->compute_batch(num_inactive, inactive_idx, dest, num_suppvec, idx, alphas);

				  j=0;
				  for(i=0;i<totdoc;i++) {
					  if(!shrink_state->active[i]) {
						  lin[i] = shrink_state->last_lin[i] + dest[j++] ;
					  }
					  shrink_state->last_lin[i]=lin[i];
				  }

				  delete[] dest;
			  }
			  else
			  {
				  for(i=0;i<totdoc;i++)
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
	  changed=new INT[totdoc];
	  changed2dnum=new INT[totdoc+11];
	  inactive=new INT[totdoc];
	  inactive2dnum=new INT[totdoc+11];
	  for(t=shrink_state->deactnum-1;(t>=0) && shrink_state->a_history[t];t--)
	  {
		  if(verbosity>=2) {
			  SG_INFO( "%ld..",t);
		  }
		  a_old=shrink_state->a_history[t];    
		  for(i=0;i<totdoc;i++) {
			  inactive[i]=((!shrink_state->active[i]) 
						   && (shrink_state->inactive_since[i] == t));
			  changed[i]= (a[i] != a_old[i]);
		  }
		  compute_index(inactive,totdoc,inactive2dnum);
		  compute_index(changed,totdoc,changed2dnum);


		  INT num_threads=parallel.get_num_threads();
		  ASSERT(num_threads>0);

		  if (num_threads < 2)
		  {
			  for(ii=0;(i=changed2dnum[ii])>=0;ii++) {
				  kernel->get_kernel_row(i,inactive2dnum,aicache);
				  for(jj=0;(j=inactive2dnum[jj])>=0;jj++)
					  lin[j]+=(a[i]-a_old[i])*aicache[j]*(double)label[i];
			  }
		  }
#ifndef WIN32
		  else
		  {
			  //find number of the changed ones
			  INT num_changed=0;
			  for(ii=0;changed2dnum[ii]>=0;ii++)
				  num_changed++;
			
			  if (num_changed>0)
			  {
				  pthread_t threads[num_threads-1];
				  S_THREAD_PARAM_REACTIVATE_VANILLA params[num_threads];
				  INT step= num_changed/num_threads;

				  // alloc num_threads many tmp buffers
				  DREAL* tmp_lin=new DREAL[totdoc*num_threads];
				  memset(tmp_lin, 0, sizeof(DREAL)*((size_t) totdoc)*num_threads);
				  DREAL* tmp_aicache=new DREAL[totdoc*num_threads];
				  memset(tmp_aicache, 0, sizeof(DREAL)*((size_t) totdoc)*num_threads);

				  INT thr;
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

				  for(jj=0;(j=inactive2dnum[jj])>=0;jj++)
					  lin[j]+=tmp_lin[totdoc*thr+j];

				  for (thr=0; thr<num_threads-1; thr++)
				  {
					  pthread_join(threads[thr], NULL);

					  //add up results
					  for(jj=0;(j=inactive2dnum[jj])>=0;jj++)
						  lin[j]+=tmp_lin[totdoc*thr+j];
				  }

				  delete[] tmp_lin;
				  delete[] tmp_aicache;
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

  if (!(kernel->has_property(KP_LINADD) && get_linadd_enabled())) { /* update history for non-linear */
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

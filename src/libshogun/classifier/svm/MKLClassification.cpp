#include "classifier/svm/MKLClassification.h"
#include "kernel/CombinedKernel.h"


CMKLClassification::CMKLClassification(CSVM* s) : CMKL(s), w_gap(1.0), rho(0)
{
}

CMKLClassification::~CMKLClassification()
{
}

struct S_THREAD_PARAM 
{
	float64_t * lin ;
	float64_t* W;
	int32_t start, end;
	int32_t * active2dnum ;
	int32_t * docs ;
	CKernel* kernel ;
};


void CMKLClassification::update_linear_component_mkl(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache)
{
	//int inner_iters=0;
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
			kn = k->get_next_kernel();
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
	
	//perform_mkl_step(beta, old_beta, num_kernels, label, active2dnum,
	//		a, lin, sumw, inner_iters);
	
	delete[] sumw;
	delete[] old_beta;
	delete[] beta;
}


void CMKLClassification::update_linear_component_mkl_linadd(
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

	if (parallel->get_num_threads() < 2)
	{
		// determine contributions of different kernels
		for (int32_t i=0; i<num; i++)
			kernel->compute_by_subkernel(i,&W[i*num_kernels]);
	}
#ifndef WIN32
	else
	{
		pthread_t* threads = new pthread_t[parallel->get_num_threads()-1];
		S_THREAD_PARAM* params = new S_THREAD_PARAM[parallel->get_num_threads()-1];
		int32_t step= num/parallel->get_num_threads();

		for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
		{
			params[t].kernel = kernel;
			params[t].W = W;
			params[t].start = t*step;
			params[t].end = (t+1)*step;
			pthread_create(&threads[t], NULL, CMKLClassification::update_linear_component_mkl_linadd_helper, (void*)&params[t]);
		}

		for (int32_t i=params[parallel->get_num_threads()-2].end; i<num; i++)
			kernel->compute_by_subkernel(i,&W[i*num_kernels]);

		for (int32_t t=0; t<parallel->get_num_threads()-1; t++)
			pthread_join(threads[t], NULL);

		delete[] params;
		delete[] threads;
	}
#endif

	// restore old weights
	kernel->set_subkernel_weights(w_backup,num_weights);

	delete[] w_backup;
	delete[] w1;

	//perform_mkl_step(beta, old_beta, num_kernels, label, active2dnum,
	//		a, lin, sumw, inner_iters);
	
	delete[] sumw;
	delete[] old_beta;
	delete[] beta;
}

void* CMKLClassification::update_linear_component_mkl_linadd_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;

	int32_t num_kernels=params->kernel->get_num_subkernels();

	// determine contributions of different kernels
	for (int32_t i=params->start; i<params->end; i++)
		params->kernel->compute_by_subkernel(i,&(params->W[i*num_kernels]));

	return NULL ;
}



void CMKLClassification::perform_mkl_step(
		float64_t* beta, const float64_t* old_beta, const float64_t* sumw,
		const float64_t suma, int32_t num_kernels, void* aux)
{
	int32_t inner_iters=0;
	float64_t mkl_objective=0;

	mkl_objective=-suma;
	for (int32_t i=0; i<num_kernels; i++)
	{
		beta[i]=old_beta[i];
		mkl_objective+=old_beta[i]*sumw[i];
	}

	w_gap = CMath::abs(1-rho/mkl_objective) ;
	if( (w_gap >= 0.9999*mkl_epsilon) || (get_solver_type()==ST_INTERNAL && mkl_norm>1) )
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
			if (get_solver_type()==ST_CPLEX) {
				rho=compute_optimal_betas_via_cplex(beta, old_beta, num_kernels, sumw, suma, inner_iters);
			}
			else
				rho=compute_optimal_betas_newton(beta, old_beta, num_kernels, sumw, suma, mkl_objective);
		}

		w_gap = CMath::abs(1-rho/mkl_objective) ;
	}
}

void CMKLClassification::set_callback_function()
{
}

float64_t CMKLClassification::compute_optimal_betas_analytically(float64_t* beta,
		const float64_t* old_beta, int32_t num_kernels,
		const float64_t* sumw, float64_t suma,
    float64_t mkl_objective)
{
	SG_DEBUG("MKL via ANALYTICAL\n");

	const float64_t r = 1.0 / ( mkl_norm - 1.0 );
	float64_t Z;
	float64_t obj;

  //obj = -suma;
	//for (int32_t d=0; d<num_kernels; d++) {
		//obj += old_beta[d] * (sumw[d]);
  //}
	//SG_PRINT( "OBJ_old = %f / %f\n", obj, mkl_objective );
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

  //
  //  // compute in log space
  //  Z = CMath::log(0.0);
  //  for (int32_t n=0; n<num_kernels; n++ )
  //  {
  //  	ASSERT(sumw[n]>=0);
  //  	beta[n] = CMath::log(sumw[n])*r;
  //  	Z = CMath::logarithmic_sum(Z, beta[n]*mkl_norm);
  //  }

  //  Z *= -1.0/mkl_norm;
  //  for (int32_t n=0; n<num_kernels; n++ )
  //  	beta[n] = CMath::exp(beta[n]+Z);
  //

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
float64_t CMKLClassification::compute_optimal_betas_gradient(float64_t* beta,
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
float64_t CMKLClassification::compute_optimal_betas_gradient(float64_t* beta,
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


 /*
float64_t CMKLClassification::compute_optimal_betas_newton(float64_t* beta,
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

  / *
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
  * /

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
      if ( 0.0 > old_beta[p] || old_beta[p] > 1.0 )
	  {
		  CMath::display_vector(old_beta, num_kernels, "old_beta");
		  CMath::display_vector(beta, num_kernels, "beta");
		  SG_ERROR("old_beta out of range");
	  }
	  float64_t t1=gamma*mkl_norm*CMath::pow(old_beta[p],mkl_norm) - sumw[p]*old_beta[p];
	  float64_t t2 = 2.0*sumw[p] + gqq1*CMath::pow(old_beta[p],mkl_norm-1.0);

	  if (t1 == 0.0)
		  beta[p]=0.0;
	  else
		  beta[p]=t1/t2;
      //beta[p] = ( gamma*mkl_norm*CMath::pow(old_beta[p],mkl_norm) - sumw[p]*old_beta[p] )
       // / ( 2.0*sumw[p] + gqq1*CMath::pow(old_beta[p],mkl_norm-1.0) );
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
        beta[p] = 1e-10;
      }
      Z += CMath::pow( beta[p], mkl_norm );
    }
    Z = CMath::pow( Z, -1.0/mkl_norm );
    for( p=0; p<num_kernels; ++p ) {
      beta[p] *= Z;
    }

  }

	//CMath::display_vector(beta, num_kernels, "beta_alex");
	//SG_PRINT("Z_alex = %e\n", Z);
	//CMath::display_vector( old_beta, num_kernels, "old_beta " );
	//CMath::display_vector( beta,     num_kernels, "beta     " );
	//CMath::display_vector(beta, num_kernels, "beta_log");
	//SG_PRINT("Z_log=%f\n", Z);
	//for (int32_t i=0; i<num_kernels; i++)
		//beta[i]=(beta[i]+old_beta[i])/2;

	CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

  obj = -suma;
	for( p=0; p<num_kernels; ++p ) {
		obj += beta[p] * (sumw[p]);
  }
	return obj;
}
*/


float64_t CMKLClassification::compute_optimal_betas_newton(float64_t* beta,
		const float64_t* old_beta, int32_t num_kernels,
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
		gamma += CMath::pow( sumw[p]*old_beta[p]*old_beta[p], r );
	}
  gamma = CMath::pow( gamma, 1.0/r ) / mkl_norm;
  ASSERT( gamma > 0.0 );

  const int nofNewtonSteps = 1;
  int i;
  if( nofNewtonSteps > 1 ) {
    SG_PRINT( "performing %d Newton steps.\n", nofNewtonSteps );
  }
  for( i = 0; i < nofNewtonSteps; ++i ) {

    //if( i != 0 ) {
    //  for( p=0; p<num_kernels; ++p ) {
    //    old_beta[p] = beta[p];
    //  }
    //}

    // compute Newton step (stored in "beta") (Hessian is diagonal)
    const float64_t gqq1 = gamma * mkl_norm * (mkl_norm-1.0);
    Z = 0.0;
    for( p=0; p<num_kernels; ++p ) {
      if ( 0.0 > old_beta[p] || old_beta[p] > 1.0 ) {
        CMath::display_vector( old_beta, num_kernels, "old_beta" );
        CMath::display_vector( beta, num_kernels, "beta" );
        SG_ERROR("old_beta out of range");
      }
      float64_t t1 = gamma*mkl_norm*CMath::pow(old_beta[p],mkl_norm) - sumw[p]*old_beta[p];
      float64_t t2 = 2.0*sumw[p] + gqq1*CMath::pow(old_beta[p],mkl_norm-1.0);
      beta[p] = ( t1 == 0.0 ) ? 0.0 : ( t1 / t2 );
      Z += beta[p] * beta[p];
    }
    //CMath::display_vector( beta, num_kernels, "newton   " );
    //SG_PRINT( "Newton step size = %e\n", Z );

    // perform Newton step
    Z = 0.0;
    for( p=0; p<num_kernels; ++p ) {
      beta[p] = old_beta[p] - beta[p];
      if( !( beta[p] >= 1e-10 ) ) {
        beta[p] = 1e-10;
      }
      Z += CMath::pow( beta[p], mkl_norm );
    }
    Z = CMath::pow( Z, -1.0/mkl_norm );
    for( p=0; p<num_kernels; ++p ) {
      beta[p] *= Z;
    }

  }

	CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);

  obj = -suma;
	for( p=0; p<num_kernels; ++p ) {
		obj += beta[p] * (sumw[p]);
  }
	return obj;
}


// float64_t CMKLClassification::compute_optimal_betas_newton(float64_t* beta,
// 		float64_t* old_beta, int32_t num_kernels,
// 		const float64_t* sumw, float64_t suma,
//     float64_t mkl_objective)
// {
// 	SG_DEBUG("MKL via NEWTON\n");
// 
// 	const float64_t r = mkl_norm / ( mkl_norm - 1.0 );
//   float64_t* log_beta = new float64_t[ num_kernels ];
// 	float64_t Z;
// 	float64_t obj;
// 	float64_t gamma;
//   int32_t p;
// 
// 	//SG_PRINT( "OBJ_old = %f\n", mkl_objective );
// 
// 	gamma = 0.0;
// 	for( p=0; p<num_kernels; ++p ) {
// 		gamma += CMath::pow( sumw[p], r );
// 	}
//   gamma = CMath::pow( gamma, 1.0/r ) / mkl_norm;
// 
//   // compute gradient (stored in "beta")
// 	for( p=0; p<num_kernels; ++p ) {
//     ASSERT( 0.0 <= old_beta[p] && old_beta[p] <= 1.0 );
// 		beta[p] = gamma * mkl_norm * CMath::pow( old_beta[p], mkl_norm-1.0 ) - sumw[p];
// 		//beta[p] *= old_beta[p];
//   }
// 
//   // compute Newton step (Hessian is diagonal)
//   const float64_t gqq1 = gamma * mkl_norm * (mkl_norm-1.0);
//   Z = 0.0;
// 	for( p=0; p<num_kernels; ++p ) {
// 		const float64_t H_pp = 2.0*sumw[p]/old_beta[p] + gqq1*CMath::pow(old_beta[p],mkl_norm-2.0);
//     if( 1 ) {
//       beta[p] /= H_pp;
//     } else {
//       beta[p] /= ( beta[p] + H_pp*old_beta[p]*old_beta[p] );
//     }
//     Z += beta[p] * beta[p];
//   }
// 	//CMath::display_vector( beta, num_kernels, "newton   " );
// 	//SG_PRINT( "Newton step size = %e\n", Z );
// 
// 	//CMath::display_vector(beta, num_kernels, "beta_alex");
// 	//SG_PRINT("Z_alex = %e\n", Z);
// 	//CMath::display_vector( old_beta, num_kernels, "old_beta " );
// 	//CMath::display_vector( beta,     num_kernels, "beta     " );
// 	//CMath::display_vector(beta, num_kernels, "beta_log");
// 	//SG_PRINT("Z_log=%f\n", Z);
// 	for( p = 0; p < num_kernels; p++ ) {
// 		beta[p] = old_beta[p] - beta[p];
// 		//beta[p] = CMath::exp( CMath::log(old_beta[p]) - beta[p] );
//     if( !( beta[p] >= 1e-10 ) ) {
//       beta[p] = 1e-10;
//     }
//   }
// 
// 	CMath::scale_vector(1/CMath::qnorm(beta, num_kernels, mkl_norm), beta, num_kernels);
// 
//   obj = -suma;
// 	for( p=0; p<num_kernels; ++p ) {
// 		obj += beta[p] * (sumw[p]);
//   }
//   delete[] log_beta;
// 	return obj;
// }


float64_t CMKLClassification::compute_optimal_betas_via_cplex(float64_t* x, const float64_t* old_beta, int32_t num_kernels,
		  const float64_t* sumw, float64_t suma, int32_t& inner_iters)
{
	/*
	SG_DEBUG("MKL via CPLEX\n");

#ifdef USE_CPLEX
	if (!lp_initialized)
	{
		SG_INFO( "creating LP\n") ;

		int32_t NUMCOLS = 2*num_kernels + 1 ;
		double   obj[NUMCOLS]; // calling external lib
		double   lb[NUMCOLS]; // calling external lib
		double   ub[NUMCOLS]; // calling external lib

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
		int initial_rmatbeg[1]; // calling external lib
		int initial_rmatind[num_kernels+1]; // calling external lib
		double initial_rmatval[num_kernels+1]; // calling external lib
		double initial_rhs[1]; // calling external lib
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
				int rmatbeg[1]; // calling external lib
				int rmatind[3]; // calling external lib
				double rmatval[3]; // calling external lib
				double rhs[1]; // calling external lib
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

		// calling external lib
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
			// 1 norm mkl
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
			// then something is wrong and we rather stop sooner than later
			rho = 1 ;
		}
	}
#else
	SG_ERROR("Cplex not enabled at compile time\n");
#endif
	return rho;
	*/
		return 0;
}

float64_t CMKLClassification::compute_optimal_betas_via_glpk(float64_t* beta, const float64_t* old_beta,
		int num_kernels, const float64_t* sumw, float64_t suma, int32_t& inner_iters)
{
	SG_DEBUG("MKL via GLPK\n");
	float64_t obj=1.0;
#ifdef USE_GLPK
	int32_t NUMCOLS = 2*num_kernels + 1 ;
	if (!lp_initialized)
	{
		SG_INFO( "creating LP\n") ;

		//set obj function, note glpk indexing is 1-based
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
		lpx_set_col_bnds(lp_glpk, NUMCOLS, LPX_FR, 0,0); //unbound

		//add first row. sum[w]=1
		int row_index = lpx_add_rows(lp_glpk, 1);
		int ind[num_kernels+2];
		float64_t val[num_kernels+2];
		for (int i=1; i<=num_kernels; i++)
		{
			ind[i] = i;
			val[i] = 1;
		}
		ind[num_kernels+1] = NUMCOLS;
		val[num_kernels+1] = 0;
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
		val[i] = -(sumw[i-1]-suma);
	}
	ind[num_kernels+1] = 2*num_kernels+1;
	val[num_kernels+1] = -1;
	lpx_set_mat_row(lp_glpk, row_index, num_kernels+1, ind, val);
	lpx_set_row_bnds(lp_glpk, row_index, LPX_UP, 0, 0);

	//optimize
	lpx_simplex(lp_glpk);
	bool res = check_lpx_status(lp_glpk);
	if (!res)
		SG_ERROR( "Failed to optimize Problem.\n");

	int32_t cur_numrows = lpx_get_num_rows(lp_glpk);
	int32_t cur_numcols = lpx_get_num_cols(lp_glpk);
	int32_t num_rows=cur_numrows;
	ASSERT(cur_numcols<=2*num_kernels+1);

	float64_t* col_primal = new float64_t[cur_numcols];
	float64_t* row_primal = new float64_t[cur_numrows];
	float64_t* row_dual = new float64_t[cur_numrows];

	for (int i=0; i<cur_numrows; i++)
	{
		row_primal[i] = lpx_get_row_prim(lp_glpk, i+1);
		row_dual[i] = lpx_get_row_dual(lp_glpk, i+1);
	}
	for (int i=0; i<cur_numcols; i++)
		col_primal[i] = lpx_get_col_prim(lp_glpk, i+1);

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
			lpx_del_rows(lp_glpk, 1, del_rows);
		}
	}

	delete[] row_dual;
	delete[] row_primal;
	delete[] col_primal;
#else
	SG_ERROR("Glpk not enabled at compile time\n");
#endif

	return obj;
}

#ifdef USE_CPLEX
void CMKLClassification::set_qnorm_constraints(float64_t* beta, int32_t num_kernels)
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

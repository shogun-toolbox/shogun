#include "classifier/svm/MKLRegression.h"
#include "kernel/CombinedKernel.h"

CMKLRegression::CMKLRegression(CSVM* s) : CMKL(s)
{
}

CMKLRegression::~CMKLRegression()
{
}

void CMKLRegression::perform_mkl_step(float64_t* alpha, float64_t* old_alpha, int32_t num_alpha,
		float64_t* beta, float64_t* old_beta, int32_t num_beta, void* aux)
{
}

void CMKLRegression::set_callback_function()
{
}

/*
void CSVRLight::update_linear_component_mkl(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache, float64_t* c)
{
	int32_t num         = totdoc;
	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	const float64_t* w  = kernel->get_subkernel_weights(num_weights);
	float64_t* beta = new float64_t[2*num_kernels+1];

	ASSERT(num_weights==num_kernels);
	float64_t* sumw=new float64_t[num_kernels];
	int32_t num_active_rows=0;
	int32_t num_rows=0;

	if ((kernel->get_kernel_type()==K_COMBINED) && 
			 (!((CCombinedKernel*)kernel)->get_append_subkernel_weights()))// for combined kernel
	{
		CCombinedKernel* k      = (CCombinedKernel*) kernel;
		CKernel* kn = k->get_first_kernel() ;
		int32_t n = 0, i, j ;
		
		while (kn!=NULL)
		{
			for(i=0;i<num;i++) 
			{
				if(a[i] != a_old[i]) 
				{
					kn->get_kernel_row(i,NULL,aicache, true);
					for(j=0;j<num;j++) 
						W[j*num_kernels+n]+=(a[i]-a_old[i])*aicache[regression_fix_index(j)]*(float64_t)label[i];
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
			w_backup[i] = w[i] ;
			w1[i]=0.0 ; 
		}
		for (int32_t n=0; n<num_kernels; n++)
		{
			w1[n]=1.0 ;
			kernel->set_subkernel_weights(w1, num_weights) ;
		
			for(int32_t i=0;i<num;i++) 
			{
				if(a[i] != a_old[i]) 
				{
					for(int32_t j=0;j<num;j++) 
						W[j*num_kernels+n]+=(a[i]-a_old[i])*kernel->kernel(regression_fix_index(i),regression_fix_index(j))*(float64_t)label[i];
				}
			}
			w1[n]=0.0 ;
		}

		// restore old weights
		kernel->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;
	}
	
	float64_t mkl_objective=0;
#ifdef HAVE_LAPACK
	int nk = (int) num_kernels; // calling external lib
	double* alphay  = new double[num];
	float64_t sumalpha = 0 ;
	
	for (int32_t i=0; i<num; i++)
	{
		alphay[i]=a[i]*label[i] ;
		sumalpha+=a[i]*(learn_parm->eps-label[i]*c[i]);
	}

	for (int32_t i=0; i<num_kernels; i++)
		sumw[i]=sumalpha ;
	
	cblas_dgemv(CblasColMajor, CblasNoTrans, nk, (int) num, 0.5, (double*) W,
		nk, alphay, 1, 1.0, (double*) sumw, 1);
	
	for (int32_t i=0; i<num_kernels; i++)
		mkl_objective+=w[i]*sumw[i] ;

	delete[] alphay;
#else
	for (int32_t d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for(int32_t i=0; i<num; i++)
			sumw[d] += a[i]*(learn_parm->eps + label[i]*(0.5*W[i*num_kernels+d]-c[i]));
		mkl_objective   += w[d]*sumw[d];
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
			
			int NUMCOLS = 2*num_kernels + 1; // calling external lib
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
			{
				obj[i]= C_mkl ;
			}
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
			int initial_rmatbeg[1]; // calling external lib
			int initial_rmatind[num_kernels+1]; // calling external lib
			double initial_rmatval[num_kernels+1]; // calling external lib
			double initial_rhs[1]; // calling external lib
			char initial_sense[1];
			
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
			if ( status ) {
				SG_ERROR( "Failed to add the first row.\n");
			}
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
					rhs[0]=0 ;     // rhs=1 ;
					sense[0]='L' ; // equality
					rmatind[0]=q ;
					rmatval[0]=1 ;
					rmatind[1]=q+1 ;
					rmatval[1]=-1 ;
					rmatind[2]=num_kernels+q ;
					rmatval[2]=-1 ;
					status = CPXaddrows (env, lp_cplex, 0, 1, 3, 
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
					status = CPXaddrows (env, lp_cplex, 0, 1, 3, 
										 rhs, sense, rmatbeg,
										 rmatind, rmatval, NULL, NULL);
					if ( status ) {
						SG_ERROR( "Failed to add a smothness row (2).\n");
					}
				}
			}
		}

		SG_DEBUG( "*") ;
		
		{ // add the new row
			//SG_INFO( "add the new row\n") ;
			
			int rmatbeg[1]; // calling external lib
			int rmatind[num_kernels+1]; // calling external lib
			double rmatval[num_kernels+1]; // calling external lib
			double rhs[1]; // calling external lib
			char sense[1];
			
			rmatbeg[0] = 0;
			rhs[0]=0 ;
			sense[0]='L' ;
			
			for (int32_t i=0; i<num_kernels; i++)
			{
				rmatind[i]=i ;
				rmatval[i]=-sumw[i] ;
			}
			rmatind[num_kernels]=2*num_kernels ;
			rmatval[num_kernels]=-1 ;
			
			int status = CPXaddrows (env, lp_cplex, 0, 1, num_kernels+1, 
									 rhs, sense, rmatbeg,
									 rmatind, rmatval, NULL, NULL);
			if ( status ) 
				SG_ERROR( "Failed to add the new row.\n");
		}
		
		{ // optimize
			int status = CPXlpopt (env, lp_cplex);
			if ( status ) 
				SG_ERROR( "Failed to optimize LP.\n");
			
			// obtain solution
			int32_t cur_numrows=(int32_t) CPXgetnumrows(env, lp_cplex);
			int32_t cur_numcols=(int32_t) CPXgetnumcols(env, lp_cplex);
			num_rows=cur_numrows;
			ASSERT(cur_numcols<=2*num_kernels+1);

			float64_t *slack=new float64_t[cur_numrows];
			float64_t *pi=new float64_t[cur_numrows];

			if (slack==NULL || pi==NULL)
			{
				status=CPXERR_NO_MEMORY;
				SG_ERROR("Could not allocate memory for solution.\n");
			}

			// calling external lib
			int solstat=0;
			float64_t objval=0;
			status=CPXsolution(env, lp_cplex, &solstat, &objval, (double*) beta,
				(double*) pi, (double*) slack, NULL);
			int32_t solution_ok=!status;
			if (status)
				SG_ERROR( "Failed to obtain solution.\n");

			num_active_rows=0 ;
			if (solution_ok)
			{
				float64_t max_slack = -CMath::INFTY ;
				int32_t max_idx = -1 ;
				int32_t start_row = 1 ;
				if (C_mkl!=0.0)
					start_row+=2*(num_kernels-1);

				for (int32_t i = start_row; i < cur_numrows; i++)  // skip first
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
					status = CPXdelrows (env, lp_cplex, max_idx, max_idx) ;
					if ( status ) 
						SG_ERROR( "Failed to remove an old row.\n");
				}

				// set weights, store new rho and compute new w gap
				kernel->set_subkernel_weights(beta, num_kernels) ;
				rho = -beta[2*num_kernels] ;
				w_gap = CMath::abs(1-rho/mkl_objective) ;
				
				delete[] pi ;
				delete[] slack ;
			} else
				w_gap = 0 ; // then something is wrong and we rather 
				            // stop sooner than later
		}
	}
#endif
	
	const float64_t* w_new   = kernel->get_subkernel_weights(num_weights);
	// update lin
#ifdef HAVE_LAPACK
	cblas_dgemv(CblasColMajor, CblasTrans, nk, (int) num, 1.0, (double*) W,
		nk, (double*) w_new, 1, 0.0, (double*) lin, 1);
#else
	for(int32_t i=0; i<num; i++)
		lin[i]=0 ;
	for (int32_t d=0; d<num_kernels; d++)
		if (w_new[d]!=0)
			for(int32_t i=0; i<num; i++)
				lin[i] += w_new[d]*W[i*num_kernels+d] ;
#endif
	
	// count actives
	int32_t jj ;
	for(jj=0;active2dnum[jj]>=0;jj++);
	
	if (count%10==0)
	{
		int32_t start_row = 1 ;
		if (C_mkl!=0.0)
			start_row+=2*(num_kernels-1);
		SG_DEBUG("\n%i. OBJ: %f  RHO: %f  wgap=%f agap=%f (activeset=%i; active rows=%i/%i)\n", count, mkl_objective,rho,w_gap,mymaxdiff,jj,num_active_rows,num_rows-start_row);
	}
	
	delete[] sumw;
	delete[] beta;
}


void CSVRLight::update_linear_component_mkl_linadd(
	int32_t* docs, int32_t* label, int32_t *active2dnum, float64_t *a,
	float64_t *a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache, float64_t* c)
{
	// kernel with LP_LINADD property is assumed to have 
	// compute_by_subkernel functions
	int32_t num         = totdoc;
	int32_t num_weights = -1;
	int32_t num_kernels = kernel->get_num_subkernels() ;
	const float64_t* w   = kernel->get_subkernel_weights(num_weights);
	int32_t num_active_rows=0;
	int32_t num_rows=0;
	float64_t* beta = new float64_t[2*num_kernels+1];
	
	ASSERT(num_weights==num_kernels);
	float64_t* sumw=new float64_t[num_kernels];
	{
		float64_t* w_backup=new float64_t[num_kernels];
		float64_t* w1=new float64_t[num_kernels];

		// backup and set to one
		for (int32_t i=0; i<num_kernels; i++)
		{
			w_backup[i] = w[i] ;
			w1[i]=1.0 ; 
		}
		// set the kernel weights
		kernel->set_subkernel_weights(w1, num_weights) ;
		
		// create normal update (with changed alphas only)
		kernel->clear_normal();
		for(int32_t ii=0, i=0;(i=working2dnum[ii])>=0;ii++) {
			if(a[i] != a_old[i]) {
				kernel->add_to_normal(regression_fix_index(docs[i]), (a[i]-a_old[i])*(float64_t)label[i]);
			}
		}
		
		// determine contributions of different kernels
		for (int32_t i=0; i<num; i++)
			kernel->compute_by_subkernel(i,&W[i*num_kernels]) ;

		// restore old weights
		kernel->set_subkernel_weights(w_backup,num_weights) ;
		
		delete[] w_backup ;
		delete[] w1 ;
	}
	float64_t mkl_objective=0;
#ifdef HAVE_LAPACK
	int nk = (int) num_kernels; // calling external lib
	float64_t sumalpha = 0 ;
	
	for (int32_t i=0; i<num; i++)
		sumalpha+=a[i]*(learn_parm->eps-label[i]*c[i]);
	
	for (int32_t i=0; i<num_kernels; i++)
		sumw[i]=-sumalpha ;
	
	cblas_dgemv(CblasColMajor, CblasNoTrans, nk, (int) num, 0.5, (double*) W,
		nk, (double*) a, 1, 1.0, (double*) sumw, 1);
	
	for (int32_t i=0; i<num_kernels; i++)
		mkl_objective+=w[i]*sumw[i] ;
#else
	for (int32_t d=0; d<num_kernels; d++)
	{
		sumw[d]=0;
		for(int32_t i=0; i<num; i++)
			sumw[d] += a[i]*(learn_parm->eps + label[i]*(0.5*W[i*num_kernels+d]-c[i]));
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
			
			int NUMCOLS = 2*num_kernels + 1; // calling external lib
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
			{
				obj[i]= C_mkl ;
			}
			obj[2*num_kernels]=1 ;
			lb[2*num_kernels]=-CPX_INFBOUND ;
			ub[2*num_kernels]=CPX_INFBOUND ;
			
			int32_t status = CPXnewcols (env, lp_cplex, NUMCOLS, obj, lb, ub, NULL, NULL);
			if ( status ) {
				char  errmsg[1024];
				CPXgeterrorstring (env, status, errmsg);
				SG_ERROR( "%s", errmsg);
			}
			
			// add constraint sum(w)=1;
			SG_INFO( "add the first row\n");
			int initial_rmatbeg[1]; // calling external lib
			int initial_rmatind[num_kernels+1]; // calling external lib
			double initial_rmatval[num_kernels+1]; // calling ext lib
			double initial_rhs[1]; // calling external lib
			char initial_sense[1];
			
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
			if ( status ) {
				SG_ERROR( "Failed to add the first row.\n");
			}
			lp_initialized=true ;
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
					rhs[0]=0 ;     // rhs=1 ;
					sense[0]='L' ; // equality
					rmatind[0]=q ;
					rmatval[0]=1 ;
					rmatind[1]=q+1 ;
					rmatval[1]=-1 ;
					rmatind[2]=num_kernels+q ;
					rmatval[2]=-1 ;
					status = CPXaddrows (env, lp_cplex, 0, 1, 3, 
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
					status = CPXaddrows (env, lp_cplex, 0, 1, 3, 
										 rhs, sense, rmatbeg,
										 rmatind, rmatval, NULL, NULL);
					if ( status ) {
						SG_ERROR( "Failed to add a smothness row (2).\n");
					}
				}
			}
		}
		
		{ // add the new row
			
			int rmatbeg[1]; // calling external lib
			int rmatind[num_kernels+1]; // calling external lib
			double rmatval[num_kernels+1]; // calling external lib
			double rhs[1]; // calling external lib
			char sense[1];
			
			rmatbeg[0] = 0;
			rhs[0]=0 ;
			sense[0]='L' ;
			
			for (int32_t i=0; i<num_kernels; i++)
			{
				rmatind[i]=i ;
				rmatval[i]=-sumw[i] ;
			}
			rmatind[num_kernels]=2*num_kernels ;
			rmatval[num_kernels]=-1 ;
			
			int32_t status = CPXaddrows (env, lp_cplex, 0, 1, num_kernels+1, 
									 rhs, sense, rmatbeg,
									 rmatind, rmatval, NULL, NULL);
			if ( status ) 
				SG_ERROR( "Failed to add the new row.\n");
		}
		
		{ // optimize
			int32_t status = CPXlpopt (env, lp_cplex);
			if ( status ) 
				SG_ERROR( "Failed to optimize LP.\n");
			
			// obtain solution
			int32_t cur_numrows=(int32_t) CPXgetnumrows(env, lp_cplex);
			int32_t cur_numcols=(int32_t) CPXgetnumcols (env, lp_cplex);
			num_rows=cur_numrows;
			ASSERT(cur_numcols<=2*num_kernels+1);
			
			float64_t* slack=new float64_t[cur_numrows];
			float64_t* pi=new float64_t[cur_numrows];

			// calling external lib
			int solstat=0;
			float64_t objval=0;
			status=CPXsolution(env, lp_cplex, &solstat, &objval, (double*) beta,
				(double*) pi, (double*) slack, NULL);
			int32_t solution_ok=!status;
			if (status)
				SG_ERROR( "Failed to obtain solution.\n");

			num_active_rows=0 ;
			if (solution_ok)
			{
				float64_t max_slack = -CMath::INFTY ;
				int32_t max_idx = -1 ;
				int32_t start_row = 1 ;
				if (C_mkl!=0.0)
					start_row+=2*(num_kernels-1);

				for (int32_t i = start_row; i < cur_numrows; i++)  // skip first
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
					status = CPXdelrows (env, lp_cplex, max_idx, max_idx) ;
					if ( status ) 
						SG_ERROR( "Failed to remove an old row.\n");
				}

				// set weights, store new rho and compute new w gap
				kernel->set_subkernel_weights(beta, num_kernels) ;
				rho = -beta[2*num_kernels] ;
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
	cblas_dgemv(CblasColMajor, CblasTrans, nk, (int) num, 1.0, (double*) W,
		nk, (double*) w, 1, 0.0, (double*) lin, 1);
#else
	for(int32_t i=0; i<num; i++)
		lin[i]=0 ;
	for (int32_t d=0; d<num_kernels; d++)
		if (w[d]!=0)
			for(int32_t i=0; i<num; i++)
				lin[i] += w[d]*W[i*num_kernels+d] ;
#endif
	
	// count actives
	int32_t jj ;
	for(jj=0;active2dnum[jj]>=0;jj++);
	
	if (count%10==0)
	{
		int32_t start_row = 1 ;
		if (C_mkl!=0.0)
			start_row+=2*(num_kernels-1);
		SG_DEBUG("\n%i. OBJ: %f  RHO: %f  wgap=%f agap=%f (activeset=%i; active rows=%i/%i)\n", count, mkl_objective,rho,w_gap,mymaxdiff,jj,num_active_rows,num_rows-start_row);
	}
	
	delete[] sumw;
	delete[] beta;
}
*/



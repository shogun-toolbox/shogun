/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#ifndef __MKLCLASSIFICATION_H__
#define __MKLCLASSIFICATION_H__

#include "lib/common.h"
#include "classifier/svm/MKL.h"

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
			pthread_create(&threads[t], NULL, CSVMLight::update_linear_component_mkl_linadd_helper, (void*)&params[t]);
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

	perform_mkl_step(beta, old_beta, num_kernels, label, active2dnum,
			a, lin, sumw, inner_iters);
	
	delete[] sumw;
	delete[] old_beta;
	delete[] beta;
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


	/** helper for update linear component MKL linadd
	 *
	 * @param p p
	 */
	static void* update_linear_component_mkl_linadd_helper(void* p);
  /** update linear component MKL
   *
   * @param docs docs
   * @param label label
   * @param active2dnum active 2D num
   * @param a a
   * @param a_old old a
   * @param working2dnum working 2D num
   * @param totdoc totdoc
   * @param lin lin
   * @param aicache ai cache
   */
  void update_linear_component_mkl(
	int32_t* docs, int32_t *label, int32_t *active2dnum, float64_t *a,
	float64_t* a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache);

  /** update linear component MKL
   *
   * @param docs docs
   * @param label label
   * @param active2dnum active 2D num
   * @param a a
   * @param a_old old a
   * @param working2dnum working 2D num
   * @param totdoc totdoc
   * @param lin lin
   * @param aicache ai cache
   */
  void update_linear_component_mkl_linadd(
	int32_t* docs, int32_t *label, int32_t *active2dnum, float64_t *a,
	float64_t* a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache);



#endif //__MKLCLASSIFICATION_H__

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#ifndef __MKL_H__
#define __MKL_H__

#ifdef USE_GLPK
#include <glpk.h>
#endif

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}
#endif

#include "lib/common.h"
#include "features/Features.h"
#include "kernel/Kernel.h"
#include "classifier/svm/SVM.h"

class CMKL : public CSVM
{
	public:
		/** Constructor
		 *
		 * @param s SVM to use as constraint generator in MKL SILP
		 */
		CMKL(CSVM* s=NULL);

		/** Destructor
		 */
		virtual ~CMKL();

		/** SVM to use as constraint generator in MKL SILP
		 *
		 * @param s svm
		 */
		void set_constraint_generator(CSVM* s)
		{
			SG_UNREF(svm);
			SG_REF(s);
			svm=s;
		}

		inline CSVM* get_svm()
		{
			SG_REF(svm);
			return svm;
		}

		inline void set_svm(CSVM* s)
		{
			ASSERT(s);
			SG_REF(s);
			SG_UNREF(svm);
			svm=s;
		}

		virtual bool train();

		/** set C mkl
		 *
		 * @param C new C_mkl
		 */
		inline void set_C_mkl(float64_t C) { C_mkl = C; }

		/** set mkl norm
		 *
		 * @param norm new mkl norm (must be greater equal 1)
		 */
		inline void set_mkl_norm(float64_t norm)
		{
			if (norm<1)
				SG_ERROR("Norm must be >= 1, e.g., 1-norm is the standard MKL; 2-norm nonsparse MKL\n");
			mkl_norm = norm;
		}

		/** set mkl epsilon (optimization accuracy for kernel weights)
		 *
		 * @param eps new weight_epsilon
		 */
		inline void set_mkl_epsilon(float64_t eps) { mkl_epsilon=eps; }

		/** get mkl epsilon for weights (optimization accuracy for kernel weights)
		 *
		 * @return epsilon for weights
		 */
		inline float64_t get_mkl_epsilon() { return mkl_epsilon; }

		/** get number of MKL iterations
		 *
		 * @return mkl_iterations
		 */
		inline int32_t get_mkl_iterations() { return mkl_iterations; }

		/** perform single mkl iteration
		 *
		 * given sum of alphas, objectives for current alphas for each kernel
		 * and current kernel weighting compute the corresponding optimal beta
		 *
		 * @param beta new betas (vector of kernel weights)
		 * @param old_beta old betas (vector of previous kernel weights)
		 * @param sumw vector of 1/2*alpha'*K_j*alpha for each kernel j
		 * @param suma scalar sum_i alpha_i
		 * @param num_kernels number of kernels
		 * @param aux auxilary storage
		 *
		 */
		virtual void perform_mkl_step(
				float64_t* beta, const float64_t* old_beta, const float64_t* sumw,
				const float64_t suma, int32_t num_kernels, void* aux)=0;

		inline float64_t compute_sum_alpha()
		{
			float64_t suma=0;
			int32_t nsv=svm->get_num_support_vectors();
			for (int32_t i=0; i<nsv; i++)
				suma+=CMath::abs(svm->get_alpha(i));

			return suma;
		}

		inline void compute_sum_beta(float64_t* sumw)
		{
			ASSERT(sumw);

			float64_t* beta = new float64_t[num_kernels];
			int32_t nweights=0;
			const float64_t* old_beta = kernel->get_subkernel_weights(nweights);
			ASSERT(nweights==num_kernels);
			ASSERT(old_beta);

			for (int32_t i=0; i<num_kernels; i++)
			{
				beta[i]=0;
				sumw[i]=0;
			}

			for (int32_t n=0; n<num_kernels; n++)
			{
				beta[n]=1.0;
				kernel->set_subkernel_weights(beta, num_kernels);

				for (int32_t i=0; i<nsv; i++)
				{   
					int32_t ii=svm->get_support_vector(i);

					for (int32_t j=0; j<nsv; j++)
					{   
						int32_t jj=svm->get_support_vector(j);
						sumw[n]+=0.5*svm->get_alpha(i)*svm->get_alpha(j)*kernel->kernel(ii,jj);
					}
				}
				beta[n]=0.0;
			}
			
			mkl_iterations++;
			kernel->set_subkernel_weights(old_beta, num_kernels);
		}

		/** assigns the callback function to the svm object
		 * */
		virtual void set_callback_function()=0;

	protected:

		void init_solver();

		virtual bool converged()=0;

#ifdef USE_CPLEX
		void set_qnorm_constraints(float64_t* beta, int32_t num_kernels);

		/** init cplex
		 *
		 * @return if init was successful
		 */
		bool init_cplex();

		/** cleanup cplex
		 *
		 * @return if cleanup was successful
		 */
		bool cleanup_cplex();
#endif

#ifdef USE_GLPK
		bool init_glpk();
		bool cleanup_glpk();
		bool check_lpx_status(LPX *lp);
#endif

	protected:
		CSVM* svm;
		/** C_mkl */
		float64_t C_mkl;
		/** norm used in mkl must be > 0 */
		float64_t mkl_norm;
		/** number of mkl steps */
		int32_t mkl_iterations;
		/** mkl_epsilon for multiple kernel learning */
		float64_t mkl_epsilon;
		/** whether to use mkl wrapper or interleaved opt. */
		bool interleaved_optimization;



#ifdef USE_CPLEX
		/** env */
		CPXENVptr     env;
		/** lp */
		CPXLPptr      lp_cplex;
#endif

#ifdef USE_GLPK
		/** lp */
		LPX* lp_glpk;
#endif
		/** if lp is initialized */
		bool lp_initialized ;
};
#endif //__MKL_H__

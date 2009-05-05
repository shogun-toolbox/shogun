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
		CMKL(CSVM* s=NULL)
			: CSVM(), svm(NULL), C_mkl(0), mkl_norm(1), 
			mkl_iterations(0), epsilon(1e-5)
	{
		set_constraint_generator(s);
#ifdef USE_CPLEX
	lp_cplex = NULL ;
	env = NULL ;
#endif

#ifdef USE_GLPK
	lp_glpk = NULL;
#endif

	lp_initialized = false ;
	}

		~CMKL()
		{
			SG_UNREF(svm);
		}

		/** SVM to use as constraint generator in MKL SILP
		 *
		 * @param C new C_mkl
		 */
		void set_constraint_generator(CSVM* s)
		{
			SG_UNREF(svm);
			SG_REF(s);
			svm=s;
		}

		bool train();

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
			if (norm<=0)
				SG_ERROR("Norm must be > 0, e.g., 1-norm is the standard MKL; 2-norm nonsparse MKL\n");
			mkl_norm = norm;
		}

		/** set epsilon for weights
		 *
		 * @param eps new weight_epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** get epsilon for weights
		 *
		 * @return epsilon for weights
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/** classify one example
		 *
		 * @param num which example to classify
		 * @return classified value
		 */
		virtual float64_t classify_example(int32_t num)
		{
			ASSERT(svm);
			return svm->classify_example(num);
		}

		/** get number of MKL iterations
		 *
		 * @return mkl_iterations
		 */
		inline int32_t get_mkl_iterations() { return mkl_iterations; }

		/** perform single mkl iteration
		 *
		 * given the alphas, compute the corresponding optimal betas
		 *
		 * @param beta new betas (kernel weights)
		 * @param old_beta old betas (previous kernel weights)
		 * @param num_kernels number of kernels
		 * @param label (from svmlight label)
		 * @param active2dnum (from svmlight active2dnum)
		 * @param a (from svmlight alphas)
		 * @param lin (from svmlight linear components)
		 * @param sumw 1/2*alpha'*K_j*alpha for each kernel j
		 * @param inner_iters number of required internal iterations
		 *
		 */
		void perform_mkl_step(float64_t* beta, float64_t* old_beta, int num_kernels,
				int32_t* label, int32_t* active2dnum,
				float64_t* a, float64_t* lin, float64_t* sumw, int32_t& inner_iters);

		/** given the alphas, compute the corresponding optimal betas
		 *
		 * @param beta new betas (kernel weights)
		 * @param old_beta old betas (previous kernel weights)
		 * @param num_kernels number of kernels
		 * @param sumw 1/2*alpha'*K_j*alpha for each kernel j
		 * @param suma (sum over alphas)
		 * @param mkl_objective the current mkl objective
		 *
		 * @return new objective value
		 */
		float64_t compute_optimal_betas_analytically(float64_t* beta, float64_t* old_beta,
				int32_t num_kernels, const float64_t* sumw, float64_t suma, float64_t mkl_objective);

		/*  float64_t compute_optimal_betas_gradient(float64_t* beta, float64_t* old_beta,
			int32_t num_kernels, const float64_t* sumw, float64_t suma, float64_t mkl_objective);
			*/

		/** given the alphas, compute the corresponding optimal betas
		 *
		 * @param beta new betas (kernel weights)
		 * @param old_beta old betas (previous kernel weights)
		 * @param num_kernels number of kernels
		 * @param sumw 1/2*alpha'*K_j*alpha for each kernel j
		 * @param suma (sum over alphas)
		 * @param mkl_objective the current mkl objective
		 *
		 * @return new objective value
		 */
		float64_t compute_optimal_betas_newton(float64_t* beta, float64_t* old_beta,
				int32_t num_kernels, const float64_t* sumw, float64_t suma, float64_t mkl_objective);

		/** given the alphas, compute the corresponding optimal betas
		 * using a lp for 1-norm mkl, a qcqp for 2-norm mkl and an
		 * iterated qcqp for general q-norm mkl.
		 *
		 * @param x new betas (kernel weights)
		 * @param old_beta old betas (previous kernel weights)
		 * @param num_kernels number of kernels
		 * @param sumw 1/2*alpha'*K_j*alpha for each kernel j
		 * @param suma (sum over alphas)
		 * @param inner_iters number of internal iterations (for statistics)
		 *
		 * @return new objective value
		 */
		float64_t compute_optimal_betas_via_cplex(float64_t* x, float64_t* old_beta, int32_t num_kernels,
				const float64_t* sumw, float64_t suma, int32_t& inner_iters);

		/** given the alphas, compute the corresponding optimal betas
		 * using a lp for 1-norm mkl
		 *
		 * @param beta new betas (kernel weights)
		 * @param old_beta old betas (previous kernel weights)
		 * @param num_kernels number of kernels
		 * @param sumw 1/2*alpha'*K_j*alpha for each kernel j
		 * @param suma (sum over alphas)
		 * @param inner_iters number of internal iterations (for statistics)
		 *
		 * @return new objective value
		 */
		float64_t compute_optimal_betas_via_glpk(float64_t* beta, float64_t* old_beta,
				int num_kernels, const float64_t* sumw, float64_t suma, int32_t& inner_iters);


		void init_solver();

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
		inline bool check_lpx_status(LPX *lp);
#endif

	protected:
		CSVM* svm;
		/** C_mkl */
		float64_t C_mkl;
		/** norm used in mkl must be > 0 */
		float64_t mkl_norm;
		/** number of mkl steps */
		int32_t mkl_iterations;
		/** epsilon for multiple kernel learning */
		float64_t epsilon;

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

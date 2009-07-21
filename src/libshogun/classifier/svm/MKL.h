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

class CSVM;
class CKernel;
class CFeatures;

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
				SG_ERROR("Norm must be >= 1, e.g., 1-norm is the standard MKL; norms>1 nonsparse MKL\n");
			mkl_norm = norm;
		}

		/** compute mkl primal objective
		 *
		 * @return computed mkl primal objective
		 */
		inline float64_t compute_mkl_primal_objective()
		{
			return compute_svm_primal_objective();
		}

		/** compute mkl dual objective
		 *
		 * @return computed dual objective
		 */
		virtual float64_t compute_mkl_dual_objective();

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
		 * and current kernel weighting compute the corresponding optimal
		 * kernel weighting (all via get/set_subkernel_weights in CCombinedKernel)
		 *
		 * @param sumw vector of 1/2*alpha'*K_j*alpha for each kernel j
		 * @param suma scalar sum_i alpha_i etc.
		 *
		 */
		virtual bool perform_mkl_step(const float64_t* sumw, float64_t suma);

		static bool perform_mkl_step_helper (CMKL* mkl,
				const float64_t* sumw, const float64_t suma)
		{
			return mkl->perform_mkl_step(sumw, suma);
		}


		virtual float64_t compute_sum_alpha()=0;
		virtual void compute_sum_beta(float64_t* sumw);

	protected:
		virtual void init_training()=0;


		void set_qnorm_constraints(float64_t* beta, int32_t num_kernels);

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
		float64_t compute_optimal_betas_analytically(float64_t* beta, const float64_t* old_beta,
				int32_t num_kernels, const float64_t* sumw, float64_t suma, float64_t mkl_objective);

		/*  float64_t compute_optimal_betas_gradient(float64_t* beta, float64_t* old_beta,
			int32_t num_kernels, const float64_t* sumw, float64_t suma, float64_t mkl_objective);
			*/

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
		float64_t compute_optimal_betas_via_cplex(float64_t* x, const float64_t* old_beta, int32_t num_kernels,
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
		float64_t compute_optimal_betas_via_glpk(float64_t* beta, const float64_t* old_beta,
				int num_kernels, const float64_t* sumw, float64_t suma, int32_t& inner_iters);

		// MKL stuff

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
		float64_t compute_optimal_betas_analytically(
				float64_t* beta, const float64_t* old_beta, const int32_t num_kernels,
				const int32_t* label, const float64_t* sumw, const float64_t suma,
				const float64_t mkl_objective);

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
		float64_t compute_optimal_betas_newton(float64_t* beta, const float64_t* old_beta,
				int32_t num_kernels, const float64_t* sumw, float64_t suma, float64_t mkl_objective);

		virtual bool converged()
		{
			return w_gap<mkl_epsilon;
		}

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

		float64_t* W;
		float64_t w_gap;
		float64_t rho;


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

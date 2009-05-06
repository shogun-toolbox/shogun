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

class CMKLClassification : public CMKL
{
	public:
		/** Constructor
		 *
		 * @param s SVM to use as constraint generator in MKL SILP
		 */
		CMKLClassification(CSVM* s=NULL) : CMKL(s) { }

		/** Destructor
		 */
		virtual ~CMKLClassification();

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
		virtual void perform_mkl_step(float64_t* beta, float64_t* old_beta, int num_kernels,
				int32_t* label, int32_t* active2dnum,
				float64_t* a, float64_t* lin, float64_t* sumw, int32_t& inner_iters)=0;

	protected:
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
};
#endif //__MKLCLASSIFICATION_H__

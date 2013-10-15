/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Ryota Tomioka (University of Tokyo)
 */
#ifndef __MKL_H__
#define __MKL_H__

#include <shogun/lib/config.h>

#ifdef USE_GLPK
#include <glpk.h>
#endif

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}
#endif

#include <shogun/lib/common.h>
#include <shogun/lib/Time.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/classifier/svm/SVM.h>

namespace shogun
{
/** @brief Multiple Kernel Learning
 *
 * A support vector machine based method for use with multiple kernels.  In
 * Multiple Kernel Learning (MKL) in addition to the SVM \f$\bf\alpha\f$ and
 * bias term \f$b\f$ the kernel weights \f$\bf\beta\f$ are estimated in
 * training. The resulting kernel method can be stated as
 *
 *  \f[
 *		f({\bf x})=\sum_{i=0}^{N-1} \alpha_i \sum_{j=0}^M \beta_j k_j({\bf x}, {\bf x_i})+b .
 *	\f]
 *
 * where \f$N\f$ is the number of training examples
 * \f$\alpha_i\f$ are the weights assigned to each training example
 * \f$\beta_j\f$ are the weights assigned to each sub-kernel
 * \f$k_j(x,x')\f$ are sub-kernels
 * and \f$b\f$ the bias.
 *
 * Kernels have to be chosen a-priori. In MKL \f$\alpha_i,\;\beta\f$ and bias are determined
 * by solving the following optimization program
 *
 * \f{eqnarray*}
 *    \mbox{min} && \gamma-\sum_{i=1}^N\alpha_i\\
 *    \mbox{w.r.t.} && \gamma\in R, \alpha\in R^N \nonumber\\
 *    \mbox{s.t.} && {\bf 0}\leq\alpha\leq{\bf 1}C,\;\;\sum_{i=1}^N \alpha_i y_i=0 \nonumber\\
 *    && \frac{1}{2}\sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j
 *       k_k({\bf x}_i,{\bf x}_j)\leq \gamma,\;\;
 *    \forall k=1,\ldots,K\nonumber\\
 * \f}
 * here C is a pre-specified regularization parameter.
 *
 * Within shogun this optimization problem is solved using semi-infinite
 * programming. For 1-norm MKL using one of the two approaches described in
 *
 * Soeren Sonnenburg, Gunnar Raetsch, Christin Schaefer, and Bernhard Schoelkopf.
 * Large Scale Multiple Kernel Learning. Journal of Machine Learning Research, 7:1531-1565, July 2006.
 *
 * The first approach (also called the wrapper algorithm) wraps around a
 * single kernel SVMs, alternatingly solving for \f$\alpha\f$ and \f$\beta\f$.
 * It is using a traditional SVM to generate new violated constraints and thus
 * requires a single kernel SVM and any of the SVMs contained in shogun
 * can be used. In the MKL step either a linear program is solved via glpk or
 * cplex or analytically or a newton (for norms>1) step is performed.
 *
 * The second much faster but also more memory demanding approach performing
 * interleaved optimization, is integrated into the chunking-based SVMlight.
 *
 * In addition sparsity of MKL can be controlled by the choice of the
 * \f$L_p\f$-norm regularizing \f$\beta\f$ as described in
 *
 * Marius Kloft, Ulf Brefeld, Soeren Sonnenburg, and Alexander Zien. Efficient
 * and accurate lp-norm multiple kernel learning. In Advances in Neural
 * Information Processing Systems 21. MIT Press, Cambridge, MA, 2009.
 *
 * An alternative way to control the sparsity is the elastic-net regularization, which can be formulated into the following optimization problem:
 * \f{eqnarray*}
 *    \mbox{min} && C\sum_{i=1}^N\ell\left(\sum_{k=1}^Kf_k(x_i)+b,y_i\right)+(1-\lambda)\left(\sum_{k=1}^K\|f_k\|_{\mathcal{H}_k}\right)^2+\lambda\sum_{k=1}^K\|f_k\|_{\mathcal{H}_k}^2\\
 *    \mbox{w.r.t.} && f_1\in\mathcal{H}_1,f_2\in\mathcal{H}_2,\ldots,f_K\in\mathcal{H}_K,\,b\in R \nonumber\\
 * \f}
 * where \f$\ell\f$ is a loss function. Here \f$\lambda\f$ controls the trade-off between the two regularization terms. \f$\lambda=0\f$ corresponds to \f$L_1\f$-MKL, whereas \f$\lambda=1\f$ corresponds to the uniform-weighted combination of kernels (\f$L_\infty\f$-MKL). This approach was studied by Shawe-Taylor (2008) "Kernel Learning for Novelty Detection" (NIPS MKL Workshop 2008) and Tomioka & Suzuki (2009) "Sparsity-accuracy trade-off in MKL" (NIPS MKL Workshop 2009).
 *
 */
class CMKL : public CSVM
{
	public:
		/** Constructor
		 *
		 * @param s SVM to use as constraint generator in MKL SIP
		 */
		CMKL(CSVM* s=NULL);

		/** Destructor
		 */
		virtual ~CMKL();

		/** SVM to use as constraint generator in MKL SIP
		 *
		 * @param s svm
		 */
		inline void set_constraint_generator(CSVM* s)
		{
			set_svm(s);
		}

		/** SVM to use as constraint generator in MKL SIP
		 *
		 * @param s svm
		 */
		inline void set_svm(CSVM* s)
		{
			SG_REF(s);
			SG_UNREF(svm);
			svm=s;
		}

		/** get SVM that is used as constraint generator in MKL SIP
		 *
		 * @return svm
		 */
		inline CSVM* get_svm()
		{
			SG_REF(svm);
			return svm;
		}

		/** set C mkl
		 *
		 * @param C new C_mkl
		 */
		inline void set_C_mkl(float64_t C) { C_mkl = C; }

		/** set mkl norm
		 *
		 * @param norm new mkl norm (must be greater equal 1)
		 */
		void set_mkl_norm(float64_t norm);

		/** set elasticnet lambda
		 *
		 * @param elasticnet_lambda new elastic net lambda (must be 0<=lambda<=1)
		 *               lambda=0: L1-MKL
		 *               lambda=1: Linfinity-MKL
		 */
		void set_elasticnet_lambda(float64_t elasticnet_lambda);

		/** set block norm q (used in block norm mkl)
		 *
		 * @param q mixed norm (1<=q<=inf)
		 */
		void set_mkl_block_norm(float64_t q);

		/** set state of optimization (interleaved or wrapper)
		 *
		 * @param enable if true interleaved optimization is used; wrapper
		 * otherwise
		 */
		inline void set_interleaved_optimization_enabled(bool enable)
		{
			interleaved_optimization=enable;
		}

		/** get state of optimization (interleaved or wrapper)
		 *
		 * @return true if interleaved optimization is used; wrapper otherwise
		 */
		inline bool get_interleaved_optimization_enabled()
		{
			return interleaved_optimization;
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


		/** compute ElasticnetMKL dual objective
		 *
		 * @return computed dual objective
		 */
		float64_t compute_elasticnet_dual_objective();

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

		/** callback helper function calling perform_mkl_step
		 *
		 * @param mkl MKL object
		 * @param sumw vector of 1/2*alpha'*K_j*alpha for each kernel j
		 * @param suma scalar sum_i alpha_i etc.
		 */
		static bool perform_mkl_step_helper (CMKL* mkl,
				const float64_t* sumw, const float64_t suma)
		{
			return mkl->perform_mkl_step(sumw, suma);
		}


		/** compute beta independent term from objective, e.g., in 2-class MKL
		 * sum_i alpha_i etc
		 */
		virtual float64_t compute_sum_alpha()=0;

		/** compute 1/2*alpha'*K_j*alpha for each kernel j (beta dependent term from objective)
		 *
		 * @param sumw vector of size num_kernels to hold the result
		 */
		virtual void compute_sum_beta(float64_t* sumw);

		/** @return object name */
		virtual const char* get_name() const { return "MKL"; }

	protected:
		/** train MKL classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** check run before starting training (to e.g. check if labeling is
		 * two-class labeling in classification case
		 */
		virtual void init_training()=0;

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
		 * using a lp for 1-norm mkl, a qcqp for 2-norm mkl and an
		 * iterated qcqp for general q-norm mkl.
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
		float64_t compute_optimal_betas_via_cplex(float64_t* beta, const float64_t* old_beta, int32_t num_kernels,
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
		float64_t compute_optimal_betas_elasticnet(
				float64_t* beta, const float64_t* old_beta, const int32_t num_kernels,
				const float64_t* sumw, const float64_t suma, const float64_t mkl_objective);

		/** helper function to compute the elastic-net sub-kernel weights */
		inline void elasticnet_transform(float64_t *beta, float64_t lmd, int32_t len)
		{
			for (int32_t i=0;i <len;i++)
				beta[i]=beta[i]/(1.0-lmd+lmd*beta[i]);
		}

		/** helper function to compute the elastic-net objective */
		void elasticnet_dual(float64_t *ff, float64_t *gg, float64_t *hh,
				const float64_t &del, const float64_t* nm, int32_t len,
				const float64_t &lambda);

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
		float64_t compute_optimal_betas_directly(
				float64_t* beta, const float64_t* old_beta, const int32_t num_kernels,
				const float64_t* sumw, const float64_t suma, const float64_t mkl_objective);

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
		float64_t compute_optimal_betas_block_norm(
				float64_t* beta, const float64_t* old_beta, const int32_t num_kernels,
				const float64_t* sumw, const float64_t suma, const float64_t mkl_objective);

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

		/** check if mkl converged, i.e. 'gap' is below epsilon
		 *
		 * @return whether mkl converged
		 */
		virtual bool converged()
		{
			return w_gap<mkl_epsilon;
		}

		/** initialize solver such as glpk or cplex */
		void init_solver();

#ifdef USE_CPLEX
		/** init cplex
		 *
		 * @return if init was successful
		 */
		bool init_cplex();

		/** set qnorm mkl constraints */
		void set_qnorm_constraints(float64_t* beta, int32_t num_kernels);

		/** cleanup cplex
		 *
		 * @return if cleanup was successful
		 */
		bool cleanup_cplex();
#endif

#ifdef USE_GLPK
		/** init glpk
		 *
		 * @return if init was successful
		 */
		bool init_glpk();

		/** cleanup glpk
		 *
		 * @return if cleanup was successful
		 */
		bool cleanup_glpk();

		/** check glpk error status
		 *
		 * @return if in good status
		 */
		bool check_glp_status(glp_prob *lp);
#endif

	protected:
		/** wrapper SVM */
		CSVM* svm;
		/** C_mkl */
		float64_t C_mkl;
		/** norm used in mkl must be > 0 */
		float64_t mkl_norm;
		/** Sparsity trade-off parameter used in ElasticnetMKL
		    must be 0<=lambda<=1
		    lambda=0: L1-MKL
		    lambda=1: Linfinity-MKL
		 */
		float64_t ent_lambda;

		/** Sparsity trade-off parameter used in block norm MKL
		 * should be 1 <= mkl_block_norm <= inf */
		float64_t mkl_block_norm;

		/** sub-kernel weights on the L1-term of ElasticnetMKL */
		float64_t* beta_local;
		/** number of mkl steps */
		int32_t mkl_iterations;
		/** mkl_epsilon for multiple kernel learning */
		float64_t mkl_epsilon;
		/** whether to use mkl wrapper or interleaved opt. */
		bool interleaved_optimization;

		/** partial objectives (one per kernel) */
		float64_t* W;

		/** gap between iterations */
		float64_t w_gap;
		/** objective after mkl iterations */
		float64_t rho;

		/** measures training time for use with get_max_train_time() */
		CTime training_time_clock;

#ifdef USE_CPLEX
		/** env */
		CPXENVptr     env;
		/** lp */
		CPXLPptr      lp_cplex;
#endif

#ifdef USE_GLPK
		/** lp */
		glp_prob* lp_glpk;

		/** lp parameters */
		glp_smcp* lp_glpk_parm;
#endif
		/** if lp is initialized */
		bool lp_initialized ;
};
}
#endif //__MKL_H__

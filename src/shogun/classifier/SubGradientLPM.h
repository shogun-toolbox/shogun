/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Written (W) 2007-2008 Vojtech Franc
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SUBGRADIENTLPM_H___
#define _SUBGRADIENTLPM_H___

#include <shogun/lib/config.h>

#ifdef USE_CPLEX
#include <shogun/lib/common.h>

#include <shogun/lib/Cplex.h>

#include <shogun/machine/LinearMachine.h>
#include <shogun/features/Features.h>
#include <shogun/features/Labels.h>

namespace shogun
{
/** @brief Class SubGradientSVM trains a linear classifier called Linear
 * Programming Machine, i.e. a SVM using a \f$\ell_1\f$ norm regularizer.
 *
 * It solves the following optimization problem using subgradient descent.
 *
 * \f{eqnarray*}
 * \min_{{\bf w}={(\bf w^+},{\bf w^-}), b, {\bf \xi}} &&
 * \sum_{i=1}^N ( {\bf w}^+_i + {\bf w}^-_i) + C \sum_{i=1}^{N} \xi_i\\
 *
 * \mbox{s.t.} && -y_i(({\bf w}^+-{\bf w}^-)^T {\bf x}_i + b)-{\bf \xi}_i \leq -1\\
 * && \quad {\bf x}_i \geq 0\\\
 * && {\bf w}_i \geq 0,\quad \forall i=1\dots N
 * \f}
 *
 * Note that this implementation is not very stable numerically for a large
 * number of dimensions. Also note that currently CPLEX is required to solve
 * this problem.
 * \sa CLPBoost
 * \sa CLPM
 */
class CSubGradientLPM : public CLinearClassifier
{
	public:
		CSubGradientLPM();
		CSubGradientLPM(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab);
		virtual ~CSubGradientLPM();

		virtual inline EClassifierType get_classifier_type() { return CT_SUBGRADIENTLPM; }

		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** set C
		 *
		 * @param c_neg new C constant for negatively labeled examples
		 * @param c_pos new C constant for positively labeled examples
		 *
		 */
		inline void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }

		inline float64_t get_C1() { return C1; }
		inline float64_t get_C2() { return C2; }

		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }
		inline bool get_bias_enabled() { return use_bias; }

		inline void set_epsilon(float64_t eps) { epsilon=eps; }
		inline float64_t get_epsilon() { return epsilon; }

		inline void set_qpsize(int32_t q) { qpsize=q; }
		inline int32_t get_qpsize() { return qpsize; }

		inline void set_qpsize_max(int32_t q) { qpsize_max=q; }
		inline int32_t get_qpsize_max() { return qpsize_max; }

	protected:
		/// returns number of changed constraints for precision work_epsilon
		/// and fills active array
		int32_t find_active(
			int32_t num_feat, int32_t num_vec, int32_t& num_active,
			int32_t& num_bound);

		/// swaps the active / old_active and computes idx_active, idx_bound
		/// and sum_CXy_active arrays and the sum_Cy_active variable
		void update_active(int32_t num_feat, int32_t num_vec);

		/// compute svm objective
		float64_t compute_objective(int32_t num_feat, int32_t num_vec);

		/// compute minimum norm subgradient
		/// return norm of minimum norm subgradient
		float64_t compute_min_subgradient(
			int32_t num_feat, int32_t num_vec, int32_t num_active,
			int32_t num_bound);

		///performs a line search to determine step size
		float64_t line_search(int32_t num_feat, int32_t num_vec);

		/// compute projection
		void compute_projection(int32_t num_feat, int32_t num_vec);

		/// only computes updates on the projection
		void update_projection(float64_t alpha, int32_t num_vec);

		/// alloc helper arrays
		void init(int32_t num_vec, int32_t num_feat);

		/// de-alloc helper arrays
		void cleanup();

		/** @return object name */
		inline virtual const char* get_name() const { return "SubGradientLPM"; }

	protected:
		float64_t C1;
		float64_t C2;
		float64_t epsilon;
		float64_t work_epsilon;
		float64_t autoselected_epsilon;
		int32_t qpsize;
		int32_t qpsize_max;
		int32_t qpsize_limit;
		bool use_bias;

		int32_t last_it_noimprovement;
		int32_t num_it_noimprovement;

		//idx vectors of length num_vec
		uint8_t* active; // 0=not active, 1=active, 2=on boundary
		uint8_t* old_active;
		int32_t* idx_active;
		int32_t* idx_bound;
		int32_t delta_active;
		int32_t delta_bound;
		float64_t* proj;
		float64_t* tmp_proj;
		int32_t* tmp_proj_idx;

		//vector of length num_feat
		float64_t* sum_CXy_active;
		float64_t* v;
		float64_t* old_v;
		float64_t sum_Cy_active;

		//vector of length num_feat
		int32_t pos_idx;
		int32_t neg_idx;
		int32_t zero_idx;
		int32_t* w_pos;
		int32_t* w_zero;
		int32_t* w_neg;
		float64_t* grad_w;
		float64_t grad_b;
		float64_t* grad_proj;
		float64_t* hinge_point;
		int32_t* hinge_idx;

		//vectors/sym matrix of size qpsize_limit
		float64_t* beta;

		CCplex* solver;
		float64_t lpmtim;
};
}
#endif //USE_CPLEX
#endif //_SUBGRADIENTLPM_H___

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

#ifndef _SUBGRADIENTSVM_H___
#define _SUBGRADIENTSVM_H___

#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Labels.h>

namespace shogun
{
/** @brief class SubGradientSVM */
class CSubGradientSVM : public CLinearMachine
{
	public:
		/** default constructor */
		CSubGradientSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CSubGradientSVM(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab);
		virtual ~CSubGradientSVM();

		/** get classifier type
		 *
		 * @return classifier type SUBGRADIENTSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_SUBGRADIENTSVM; }

		/** train SVM classifier
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
		 * @param c_neg C1
		 * @param c_pos C2
		 */
		inline void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }


		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline float64_t get_C2() { return C2; }

		/** set if bias shall be enabled
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** check if bias is enabled
		 *
		 * @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/** set qpsize
		 *
		 * @param q new qpsize
		 */
		inline void set_qpsize(int32_t q) { qpsize=q; }

		/** get qpsize
		 *
		 * @return qpsize
		 */
		inline int32_t get_qpsize() { return qpsize; }

		/** set qpsize_max
		 *
		 * @param q new qpsize_max
		 */
		inline void set_qpsize_max(int32_t q) { qpsize_max=q; }

		/** get qpsize_max
		 *
		 * @return qpsize_max
		 */
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
		inline virtual const char* get_name() const { return "SubGradientSVM"; }

	protected:
		/** C1 */
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** epsilon */
		float64_t epsilon;
		/** work epsilon */
		float64_t work_epsilon;
		/** autoselected epsilon */
		float64_t autoselected_epsilon;
		/** qpsize */
		int32_t qpsize;
		/** maximum qpsize */
		int32_t qpsize_max;
		/** limit of qpsize */
		int32_t qpsize_limit;
		/** shall bias be used */
		bool use_bias;

		/** last iteration no improvement */
		int32_t last_it_noimprovement;
		/** number of iterations no improvement */
		int32_t num_it_noimprovement;

		//idx vectors of length num_vec
		/** 0=not active, 1=active, 2=on boundary */
		uint8_t* active;
		/** old active */
		uint8_t* old_active;
		/** idx active */
		int32_t* idx_active;
		/** idx bound */
		int32_t* idx_bound;
		/** delta active */
		int32_t delta_active;
		/** delta bound */
		int32_t delta_bound;
		/** proj */
		float64_t* proj;
		/** tmp proj*/
		float64_t* tmp_proj;
		/** tmp proj index */
		int32_t* tmp_proj_idx;

		//vector of length num_feat
		/** sum CXy active */
		float64_t* sum_CXy_active;
		/** v */
		float64_t* v;
		/** old v */
		float64_t* old_v;
		/** sum Cy active */
		float64_t sum_Cy_active;

		//vector of length num_feat
		/** grad w */
		float64_t* grad_w;
		/** grad b */
		float64_t grad_b;
		/** grad proj */
		float64_t* grad_proj;
		/** hinge point */
		float64_t* hinge_point;
		/** hinge index */
		int32_t* hinge_idx;

		//vectors/sym matrix of size qpsize_limit
		/** beta */
		float64_t* beta;
		/** old beta */
		float64_t* old_beta;
		/** Zv */
		float64_t* Zv;
		/** old Zv */
		float64_t* old_Zv;
		/** Z */
		float64_t* Z;
		/** old Z */
		float64_t* old_Z;

		/** timing measurement */
		float64_t tim;
};
}
#endif

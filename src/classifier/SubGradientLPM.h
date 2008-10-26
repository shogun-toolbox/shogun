/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Written (W) 2007-2008 Vojtech Franc
 * Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SUBGRADIENTLPM_H___
#define _SUBGRADIENTLPM_H___

#include "lib/config.h"

#ifdef USE_CPLEX
#include "lib/common.h"

#include "lib/Cplex.h"

#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

class CSubGradientLPM : public CSparseLinearClassifier
{
	public:
		CSubGradientLPM();
		CSubGradientLPM(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);
		virtual ~CSubGradientLPM();

		virtual inline EClassifierType get_classifier_type() { return CT_SUBGRADIENTLPM; }
		virtual bool train();

		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }
		inline bool get_bias_enabled() { return use_bias; }

		inline void set_epsilon(DREAL eps) { epsilon=eps; }
		inline DREAL get_epsilon() { return epsilon; }

		inline void set_qpsize(int32_t q) { qpsize=q; }
		inline int32_t get_qpsize() { return qpsize; }

		inline void set_qpsize_max(int32_t q) { qpsize_max=q; }
		inline int32_t get_qpsize_max() { return qpsize_max; }

	protected:
		/// returns number of changed constraints for precision work_epsilon
		/// and fills active array
		int32_t find_active(int32_t num_feat, int32_t num_vec, int32_t& num_active, int32_t& num_bound);

		/// swaps the active / old_active and computes idx_active, idx_bound
		/// and sum_CXy_active arrays and the sum_Cy_active variable
		void update_active(int32_t num_feat, int32_t num_vec);

		/// compute svm objective
		DREAL compute_objective(int32_t num_feat, int32_t num_vec);

		/// compute minimum norm subgradient
		/// return norm of minimum norm subgradient
		DREAL compute_min_subgradient(int32_t num_feat, int32_t num_vec, int32_t num_active, int32_t num_bound);

		///performs a line search to determine step size
		DREAL line_search(int32_t num_feat, int32_t num_vec);

		/// compute projection
		void compute_projection(int32_t num_feat, int32_t num_vec);

		/// only computes updates on the projection
		void update_projection(DREAL alpha, int32_t num_vec);

		/// alloc helper arrays
		void init(int32_t num_vec, int32_t num_feat);
		
		/// de-alloc helper arrays
		void cleanup();

	protected:
		DREAL C1;
		DREAL C2;
		DREAL epsilon;
		DREAL work_epsilon;
		DREAL autoselected_epsilon;
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
		DREAL* proj;
		DREAL* tmp_proj;
		int32_t* tmp_proj_idx;
		
		//vector of length num_feat
		DREAL* sum_CXy_active;
		DREAL* v;
		DREAL* old_v;
		DREAL sum_Cy_active;

		//vector of length num_feat
		int32_t pos_idx;
		int32_t neg_idx;
		int32_t zero_idx;
		int32_t* w_pos;
		int32_t* w_zero;
		int32_t* w_neg;
		DREAL* grad_w;
		DREAL grad_b;
		DREAL* grad_proj;
		DREAL* hinge_point;
		int32_t* hinge_idx;

		//vectors/sym matrix of size qpsize_limit
		DREAL* beta;

		CCplex* solver;
};
#endif //USE_CPLEX
#endif //_SUBGRADIENTLPM_H___

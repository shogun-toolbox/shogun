/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Written (W) 2007 Vojtech Franc 
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SUBGRADIENTLPM_H___
#define _SUBGRADIENTLPM_H___

#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

class CSubGradientLPM : public CSparseLinearClassifier
{
	public:
		CSubGradientLPM();
		CSubGradientLPM(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);
		virtual ~CSubGradientLPM();

		inline EClassifierType get_classifier_type() { return CT_SUBGRADIENTLPM; }
		virtual bool train();

		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }
		inline bool get_bias_enabled() { return use_bias; }

		inline void set_epsilon(DREAL eps) { epsilon=eps; }
		inline DREAL get_epsilon() { return epsilon; }

		inline void set_qpsize(INT q) { qpsize=q; }
		inline INT get_qpsize() { return qpsize; }

		inline void set_qpsize_max(INT q) { qpsize_max=q; }
		inline INT get_qpsize_max() { return qpsize_max; }

	protected:
		/// returns number of changed constraints for precision work_epsilon
		/// and fills active array
		INT find_active(INT num_feat, INT num_vec, INT& num_active, INT& num_bound);

		/// swaps the active / old_active and computes idx_active, idx_bound
		/// and sum_CXy_active arrays and the sum_Cy_active variable
		void update_active(INT num_feat, INT num_vec);

		/// compute svm objective
		DREAL compute_objective(INT num_feat, INT num_vec);

		/// compute minimum norm subgradient
		/// return norm of minimum norm subgradient
		DREAL compute_min_subgradient(INT num_feat, INT num_vec, INT num_active, INT num_bound);

		///performs a line search to determine step size
		DREAL line_search(INT num_feat, INT num_vec);

		/// compute projection
		void compute_projection(INT num_feat, INT num_vec);

		/// only computes updates on the projection
		void update_projection(DREAL alpha, INT num_vec);

		/// alloc helper arrays
		void init(INT num_vec, INT num_feat);
		
		/// de-alloc helper arrays
		void cleanup();

	protected:
		DREAL C1;
		DREAL C2;
		DREAL epsilon;
		DREAL work_epsilon;
		DREAL autoselected_epsilon;
		INT qpsize;
		INT qpsize_max;
		INT qpsize_limit;
		bool use_bias;

		INT last_it_noimprovement;
		INT num_it_noimprovement;

		//idx vectors of length num_vec
		BYTE* active; // 0=not active, 1=active, 2=on boundary
		BYTE* old_active;
		INT* idx_active;
		INT* idx_bound;
		INT delta_active;
		INT delta_bound;
		DREAL* proj;
		DREAL* tmp_proj;
		INT* tmp_proj_idx;
		
		//vector of length num_feat
		DREAL* sum_CXy_active;
		DREAL* v;
		DREAL* old_v;
		DREAL sum_Cy_active;

		//vector of length num_feat
		DREAL* grad_w;
		DREAL grad_b;
		DREAL* grad_proj;
		DREAL* hinge_point;
		INT* hinge_idx;

		//vectors/sym matrix of size qpsize_limit
		DREAL* beta;
		DREAL* old_beta;
		DREAL* Zv;
		DREAL* old_Zv;
		DREAL* Z;
		DREAL* old_Z;
};
#endif


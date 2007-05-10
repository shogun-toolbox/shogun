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

#ifndef _SUBGRADIENTSVM_H___
#define _SUBGRADIENTSVM_H___

#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

class CSubGradientSVM : public CSparseLinearClassifier
{
	public:
		CSubGradientSVM();
		CSubGradientSVM(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);
		virtual ~CSubGradientSVM();

		inline EClassifierType get_classifier_type() { return CT_SUBGRADIENTSVM; }
		virtual bool train();

		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

		inline void set_epsilon(DREAL eps) { epsilon=eps; }
		inline DREAL get_epsilon() { return epsilon; }

	protected:
		/// updates the active / old_active as well as the idx_active, idx_bound
		/// and sum_CXy_active arrays and the sum_Cy_active variable
		/// returns number of changed constraints
		INT find_active(INT num_feat, INT num_vec, INT& num_active, INT& num_bound);

		/// compute svm objective
		DREAL compute_objective(INT num_feat, INT num_vec);

		/// compute minimum norm subgradient
		/// return norm of minimum norm subgradient
		DREAL compute_min_subgradient(INT num_feat, INT num_vec, INT num_active, INT num_bound);

		///performs a line search to determine step size
		DREAL line_search();

		/// update projection
		void update_projection(INT num_feat, INT num_vec);

		/// alloc helper arrays
		void init(INT num_vec, INT num_feat);
		
		/// de-alloc helper arrays
		void cleanup();

	protected:
		DREAL C1;
		DREAL C2;
		DREAL epsilon;

		//idx vectors of length num_vec
		BYTE* active; // 0=not active, 1=active, 2=on boundary
		BYTE* old_active;
		INT* idx_active;
		INT* idx_bound;
		DREAL* proj;
		
		//vector of length num_feat
		DREAL* sum_CXy_active;
		DREAL sum_Cy_active;

		//vector of length num_feat
		DREAL* grad_w;
		DREAL grad_b;

		DREAL bias;
};
#endif


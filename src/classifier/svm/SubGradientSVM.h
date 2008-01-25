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

#ifndef _SUBGRADIENTSVM_H___
#define _SUBGRADIENTSVM_H___

#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

/** class SubGradientSVM */
class CSubGradientSVM : public CSparseLinearClassifier
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
		CSubGradientSVM(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);
		virtual ~CSubGradientSVM();

		/** get classifier type
		 *
		 * @return classifier type SUBGRADIENTSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_SUBGRADIENTSVM; }

		/** train SVM
		 *
		 * @return if training was successful
		 */
		virtual bool train();

		/** set C
		 *
		 * @param c1 new C1
		 * @param c2 new C2
		 */
		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		/** get C1
		 *
		 * @return C1
		 */
		inline DREAL get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline DREAL get_C2() { return C2; }

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
		inline void set_epsilon(DREAL eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline DREAL get_epsilon() { return epsilon; }

		/** set qpsize
		 *
		 * @param q new qpsize
		 */
		inline void set_qpsize(INT q) { qpsize=q; }

		/** get qpsize
		 *
		 * @return qpsize
		 */
		inline INT get_qpsize() { return qpsize; }

		/** set qpsize_max
		 *
		 * @param q new qpsize_max
		 */
		inline void set_qpsize_max(INT q) { qpsize_max=q; }

		/** get qpsize_max
		 *
		 * @return qpsize_max
		 */
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
		/** C1 */
		DREAL C1;
		/** C2 */
		DREAL C2;
		/** epsilon */
		DREAL epsilon;
		/** work epsilon */
		DREAL work_epsilon;
		/** autoselected epsilon */
		DREAL autoselected_epsilon;
		/** qpsize */
		INT qpsize;
		/** maximum qpsize */
		INT qpsize_max;
		/** limit of qpsize */
		INT qpsize_limit;
		/** shall bias be used */
		bool use_bias;

		/** last iteration no improvement */
		INT last_it_noimprovement;
		/** number of iterations no improvement */
		INT num_it_noimprovement;

		//idx vectors of length num_vec
		/** 0=not active, 1=active, 2=on boundary */
		BYTE* active;
		/** old active */
		BYTE* old_active;
		/** idx active */
		INT* idx_active;
		/** idx bound */
		INT* idx_bound;
		/** delta active */
		INT delta_active;
		/** delta bound */
		INT delta_bound;
		/** proj */
		DREAL* proj;
		/** tmp proj*/
		DREAL* tmp_proj;
		/** tmp proj index */
		INT* tmp_proj_idx;
		
		//vector of length num_feat
		/** sum CXy active */
		DREAL* sum_CXy_active;
		/** v */
		DREAL* v;
		/** old v */
		DREAL* old_v;
		/** sum Cy active */
		DREAL sum_Cy_active;

		//vector of length num_feat
		/** grad w */
		DREAL* grad_w;
		/** grad b */
		DREAL grad_b;
		/** grad proj */
		DREAL* grad_proj;
		/** hinge point */
		DREAL* hinge_point;
		/** hinge index */
		INT* hinge_idx;

		//vectors/sym matrix of size qpsize_limit
		/** beta */
		DREAL* beta;
		/** old beta */
		DREAL* old_beta;
		/** Zv */
		DREAL* Zv;
		/** old Zv */
		DREAL* old_Zv;
		/** Z */
		DREAL* Z;
		/** old Z */
		DREAL* old_Z;
};
#endif


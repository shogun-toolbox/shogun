/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2010 Soeren Sonnenburg
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
 * Copyright (C) 2007-2010 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBLINEAR_H___
#define _LIBLINEAR_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/classifier/svm/SVM_linear.h>

namespace shogun
{
	/** liblinar solver type */
	enum LIBLINEAR_SOLVER_TYPE
	{
		/// L2 regularized linear logistic regression
		L2R_LR,
		/// L2 regularized SVM with L2-loss using dual coordinate descent
		L2R_L2LOSS_SVC_DUAL,
		/// L2 regularized SVM with L2-loss using newton in the primal
		L2R_L2LOSS_SVC,
		/// L2 regularized linear SVM with L1-loss using dual coordinate descent
		// (default since this is the standard SVM)
		L2R_L1LOSS_SVC_DUAL,
		/// linear multi-class svm by Crammer and Singer
		MCSVM_CS,
		/// L1 regularized SVM with L2-loss using dual coordinate descent
		L1R_L2LOSS_SVC,
		/// L1 regularized logistic regression
		L1R_LR
	};

#ifdef HAVE_LAPACK

/** @brief class to implement LibLinear */
class CLibLinear : public CLinearMachine
{
	public:
		/** default constructor  */
		CLibLinear(void);

		/** constructor
		 *
		 * @param liblinear_solver_type liblinear_solver_type
		 */
		CLibLinear(LIBLINEAR_SOLVER_TYPE liblinear_solver_type);

		/** constructor (using L2R_L1LOSS_SVC_DUAL as default)
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab training labels
		 */
		CLibLinear(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab);

		/** destructor */
		virtual ~CLibLinear();

		inline LIBLINEAR_SOLVER_TYPE get_liblinear_solver_type()
		{
			return liblinear_solver_type;
		}

		inline void set_liblinear_solver_type(LIBLINEAR_SOLVER_TYPE st)
		{
			liblinear_solver_type=st;
		}

		/** get classifier type
		 *
		 * @return the classifier type
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIBLINEAR; }

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

		/** @return object name */
		inline virtual const char* get_name() const { return "LibLinear"; }

		/** get the maximum number of iterations liblinear is allowed to do */
		inline int32_t get_max_iterations()
		{
			return max_iterations;
		}

		/** set the maximum number of iterations liblinear is allowed to do */
		inline void set_max_iterations(int32_t max_iter=1000)
		{
			max_iterations=max_iter;
		}

		/** set the linear term for qp */
		inline void set_linear_term(SGVector<float64_t> linear_term)
		{
			if (!labels)
				SG_ERROR("Please assign labels first!\n");

			int32_t num_labels=labels->get_num_labels();

			if (num_labels!=linear_term.vlen)
			{
				SG_ERROR("Number of labels (%d) does not match number"
						" of entries (%d) in linear term \n", num_labels,
						linear_term.vlen);
			}

			m_linear_term.destroy_vector();
			m_linear_term.vector=CMath::clone_vector(linear_term.vector,
				linear_term.vlen);
			m_linear_term.vlen=linear_term.vlen;
		}

		/** get the linear term for qp */
		SGVector<float64_t> get_linear_term();

		/** set the linear term for qp */
		void init_linear_term();

	protected:
		/** train linear SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	private:
		/** set up parameters */
        void init();

		void train_one(const problem *prob, const parameter *param, double Cp, double Cn);
		void solve_l2r_l1l2_svc(
			const problem *prob, double eps, double Cp, double Cn, LIBLINEAR_SOLVER_TYPE st);

		void solve_l1r_l2_svc(problem *prob_col, double eps, double Cp, double Cn);
		void solve_l1r_lr(const problem *prob_col, double eps, double Cp, double Cn);


	protected:
		/** C1 */
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** if bias shall be used */
		bool use_bias;
		/** epsilon */
		float64_t epsilon;
		/** maximum number of iterations */
		int32_t max_iterations;

		/** precomputed linear term */
		SGVector<float64_t> m_linear_term;

		/** solver type */
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type;
};

#endif //HAVE_LAPACK

} /* namespace shogun  */

#endif //_LIBLINEAR_H___

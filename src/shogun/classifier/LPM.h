/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Heiko Strathmann, 
 *          Saurabh Goyal, Sergey Lisitsyn
 */

#ifndef _LPM_H___
#define _LPM_H___

#include <shogun/lib/config.h>
#ifdef USE_CPLEX

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
/** @brief Class LPM trains a linear classifier called Linear Programming
 * Machine, i.e. a SVM using a \f$\ell_1\f$ norm regularizer.
 *
 * It solves the following optimization problem using CPLEX:
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
 * Note that currently CPLEX is required to solve this problem. A
 * faster implementation is available in CLPBoost.
 *
 * \sa CLPBoost
 */
class CLPM : public LinearMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		CLPM();
		virtual ~CLPM();

		virtual EMachineType get_classifier_type()
		{
			return CT_LPM;
		}

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(DotFeatures* feat)
		{
			if (feat->get_feature_class() != C_SPARSE ||
				feat->get_feature_type() != F_DREAL)
				SG_ERROR("LPM requires SPARSE REAL valued features\n")

			LinearMachine::set_features(feat);
		}

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

		/** @return object name */
		virtual const char* get_name() const { return "LPM"; }

	protected:
		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(Features* data=NULL);

	protected:
		float64_t C1;
		float64_t C2;
		bool use_bias;
		float64_t epsilon;
};
}
#endif //USE_CPLEX
#endif //_LPM_H___

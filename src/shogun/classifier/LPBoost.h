/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Heiko Strathmann, 
 *          Saurabh Goyal, Sergey Lisitsyn
 */

#ifndef _LPBOOST_H___
#define _LPBOOST_H___

#include <shogun/lib/config.h>
#ifdef USE_CPLEX

#include <shogun/lib/common.h>
#include <shogun/lib/DynamicArray.h>

#include <shogun/features/Features.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
/** @brief Class LPBoost trains a linear classifier called Linear Programming
 * Machine, i.e. a SVM using a \f$\ell_1\f$ norm regularizer.
 *
 * It solves the following optimization problem using Boosting on the input
 * features:
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
 * Note that currently CPLEX is required to solve this problem. This
 * implementation is faster than solving the linear program directly in CPLEX
 * (as was done in CLPM).
 *
 * \sa CLPM
 */
class CLPBoost : public LinearMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		CLPBoost();
		virtual ~CLPBoost();

		virtual EMachineType get_classifier_type()
		{
			return CT_LPBOOST;
		}

		bool init(int32_t num_vec);
		void cleanup();

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(DotFeatures* feat)
		{
			if (feat->get_feature_class() != C_SPARSE ||
				feat->get_feature_type() != F_DREAL)
				SG_ERROR("LPBoost requires SPARSE REAL valued features\n")

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

		float64_t find_max_violator(int32_t& max_dim);

		/** @return object name */
		virtual const char* get_name() const { return "LPBoost"; }

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

		float64_t* u;
		DynamicArray<int32_t>* dim;

		int32_t num_sfeat;
		int32_t num_svec;
		SGSparseVector<float64_t>* sfeat;

};
}
#endif //USE_CPLEX
#endif //_LPBOOST_H___

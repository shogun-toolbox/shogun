/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Harshit Syal, Fernando Iglesias, Bjoern Esser,
 *          Sergey Lisitsyn
 */

#ifndef _NEWTONSVM_H___
#define _NEWTONSVM_H___

#include <shogun/lib/config.h>

#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/common.h>
#include <shogun/machine/IterativeMachine.h>

namespace shogun
{
/** @brief NewtonSVM,
 *  In this Implementation linear SVM is trained in its primal form using Newton-like iterations.
 *  This Implementation is ported from the Olivier Chapelles fast newton based SVM solver, Which could be found here :http://mloss.org/software/view/30/
 *  For further information on this implementation of SVM refer to this paper: http://www.kyb.mpg.de/publications/attachments/neco_%5B0%5D.pdf
*/
class NewtonSVM : public IterativeMachine<LinearMachine>
{
	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor */
		NewtonSVM();

		/** constructor
		 * @param C constant C
		 * @param itr constant no of iterations
		 */
		NewtonSVM(float64_t C, int32_t itr = 20);

		~NewtonSVM() override;

		/** get classifier type
		 *
		 * @return classifier type NewtonSVM
		 */
		EMachineType get_classifier_type() override { return CT_NEWTONSVM; }

		/**
		 * set C
		 * @param C constant C
		 */
		inline void set_C(float64_t c) { C=c; }

		/** get epsilon
		 *  @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/**
		 * set epsilon
		 * @param epsilon constant epsilon
		 */
		inline void set_epsilon(float64_t e) { epsilon=e; }

		/** get C
		 *  @return C
		 */
		inline float64_t get_C() { return C; }


		/** set if bias shall be enabled
		 *  @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** get if bias is enabled
		 *  @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** set num_iter
		 *  @return num_iter
		 */
		inline int32_t get_num_iter() {return num_iter;}

		/** set iter
		 *  @param num_iter number of iterations
		 */
		inline void set_num_iter(int32_t iter) { num_iter=iter; }

		/** @return object name */
		const char* get_name() const override { return "NewtonSVM"; }

	protected:
		void init_model(const std::shared_ptr<DotFeatures>& data) override;
		virtual void iteration(
		    const std::shared_ptr<DotFeatures>& data,
		    const std::shared_ptr<Labels>& labs) override;

	private:
		void obj_fun_linear(
		    const std::shared_ptr<DotFeatures>& data,
		    const std::shared_ptr<Labels>& labs);

		void line_search_linear(
		    const SGVector<float64_t>& d, const std::shared_ptr<DotFeatures>& data,
		    const std::shared_ptr<Labels>& labs);

	protected:
		/** lambda=1/C */
		float64_t lambda, C, epsilon;
		SGVector<float64_t> out, grad;
		SGVector<int32_t> sv;
		float64_t prec, obj, t;
		int32_t x_n, x_d, num_iter, size_sv;

		/** if bias is used */
		bool use_bias;
};
}
#endif //_NEWTONSVM_H___

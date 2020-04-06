/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Fernando Iglesias, Yuyu Zhang, 
 *          Evan Shelhamer
 */
#ifndef __MKLREGRESSION_H__
#define __MKLREGRESSION_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/classifier/mkl/MKL.h>

namespace shogun
{
/** @brief Multiple Kernel Learning for regression
 *
 * Performs support vector regression while learning kernel weights at the same
 * time. Makes only sense if multiple kernels are used.
 *
 * \sa MKL
 */
class MKLRegression : public MKL
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** Constructor
		 *
		 * @param s SVM to use as constraint generator in MKL SILP
		 */
		MKLRegression(std::shared_ptr<SVM> s=NULL);

		/** Destructor
		 */
		~MKLRegression() override;

		/** compute beta independent term from objective, e.g., in 2-class MKL
		 * sum_i alpha_i etc
		 */
		float64_t compute_sum_alpha() override;

		/** @return object name */
		const char* get_name() const override { return "MKLRegression"; }

	protected:
		/** check run before starting training (to e.g. check if labeling is
		 * two-class labeling in classification case
		 */
		void init_training() override;

		/** get classifier type
		 *
		 * @return classifier type MKL_REGRESSION
		 */
		EMachineType get_classifier_type() override { return CT_MKLREGRESSION; }

		/** compute mkl dual objective
		 *
		 * @return computed dual objective
		 */
		float64_t compute_mkl_dual_objective() override;
};
}
#endif //__MKLREGRESSION_H__

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn
 */
#ifndef __MKLONECLASS_H__
#define __MKLONECLASS_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/classifier/mkl/MKL.h>

namespace shogun
{
/** @brief Multiple Kernel Learning for one-class-classification
 *
 * Learns a One-Class SVM classifier and its kernel weights. Makes only sense
 * if multiple kernels are used.
 *
 * \sa MKL
 */
class MKLOneClass : public MKL
{
	public:
		/** Constructor
		 *
		 * @param s SVM to use as constraint generator in MKL SILP
		 */
		MKLOneClass(const std::shared_ptr<SVM>& s=NULL);

		/** Destructor
		 */
		virtual ~MKLOneClass();

		/** compute beta independent term from objective, e.g., in 2-class MKL
		 * sum_i alpha_i etc
		 */
		virtual float64_t compute_sum_alpha();

		/** @return object name */
		virtual const char* get_name() const { return "MKLOneClass"; }

	protected:
		/** check run before starting training (to e.g. check if labeling is
		 * two-class labeling in classification case
		 */
		virtual void init_training();

		/** get classifier type
		 *
		 * @return classifier type MKL ONECLASS
		 */
		virtual EMachineType get_classifier_type() { return CT_MKLONECLASS; }
};
}
#endif //__MKLONECLASS_H__

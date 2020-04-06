/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni, Evan Shelhamer,
 *          Sergey Lisitsyn
 */
#ifndef __MKLCLASSIFICATION_H__
#define __MKLCLASSIFICATION_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/classifier/mkl/MKL.h>

namespace shogun
{
/** @brief Multiple Kernel Learning for two-class-classification
 *
 * Learns an SVM classifier and its kernel weights. Makes only sense if
 * multiple kernels are used.
 *
 * \sa MKL
 */
class MKLClassification : public MKL
{
	public:
		/** Constructor
		 *
		 * @param s SVM to use as constraint generator in MKL SILP
		 */
		MKLClassification(std::shared_ptr<SVM> s=NULL);

		/** Destructor
		 */
		~MKLClassification() override;

		/** compute beta independent term from objective, e.g., in 2-class MKL
		 * sum_i alpha_i etc
		 */
		float64_t compute_sum_alpha() override;

		/**
		 * Helper method used to specialize a base class instance.
		 * @param machine the machine we want to cast
		 * @return a MKLClassification machine (already SGREF'ed)
		 */
#ifndef SWIG
		[[deprecated("use .as template function")]]
#endif
		static std::shared_ptr<MKLClassification> obtain_from_generic(const std::shared_ptr<Machine>& machine);

		/** @return object name */
		const char* get_name() const override { return "MKLClassification"; }

	protected:
		/** check run before starting training (to e.g. check if labeling is
		 * two-class labeling in classification case
		 */
		void init_training() override;

		/** get classifier type
		 *
		 * @return classifier type MKL_CLASSIFICATION
		 */
		EMachineType get_classifier_type() override { return CT_MKLCLASSIFICATION; }
};
}
#endif //__MKLCLASSIFICATION_H__

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vojtech Franc, Soeren Sonnenburg, Heiko Strathmann, Evan Shelhamer, 
 *          Sergey Lisitsyn
 */

#ifndef _GNPPSVM_H___
#define _GNPPSVM_H___

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>

namespace shogun
{
/** @brief class GNPPSVM */
class GNPPSVM : public SVM
{
	public:
		/** default constructor */
		GNPPSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		GNPPSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab);

		~GNPPSVM() override;

		/** get classifier type
		 *
		 * @return classifier type GNPPSVM
		 */
		EMachineType get_classifier_type() override { return CT_GNPPSVM; }

		/** @return object name */
		const char* get_name() const override { return "GNPPSVM"; }

	protected:
		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		bool train_machine(std::shared_ptr<Features> data=NULL) override;
};
}
#endif

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, Saurabh Goyal, 
 *          Leon Kuchenbecker
 */

#ifndef _LIBSVM_ONECLASS_H___
#define _LIBSVM_ONECLASS_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/external/shogun_libsvm.h>


namespace shogun
{
/** @brief class LibSVMOneClass */
class LibSVMOneClass : public SVM
{
	public:
		/** default constructor */
		LibSVMOneClass();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 */
		LibSVMOneClass(float64_t C, std::shared_ptr<Kernel> k);
		~LibSVMOneClass() override;

		/** get classifier type
		 *
		 * @return classifier type LIBSVMONECLASS
		 */
		EMachineType get_classifier_type() override { return CT_LIBSVMONECLASS; }

		/** @return object name */
		const char* get_name() const override { return "LibSVMOneClass"; }

		bool train_require_labels() const override { return false; }

	protected:
		/** train SVM
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

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
class CLibSVMOneClass : public CSVM
{
	public:
		/** default constructor */
		CLibSVMOneClass();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 */
		CLibSVMOneClass(float64_t C, CKernel* k);
		virtual ~CLibSVMOneClass();

		/** get classifier type
		 *
		 * @return classifier type LIBSVMONECLASS
		 */
		virtual EMachineType get_classifier_type() { return CT_LIBSVMONECLASS; }

		/** @return object name */
		virtual const char* get_name() const { return "LibSVMOneClass"; }

	protected:

		virtual bool train_require_labels() const { return false; }

		/** train SVM
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);
};
}
#endif

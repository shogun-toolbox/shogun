/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Yuyu Zhang, Leon Kuchenbecker
 */

#ifndef _LIBSVM_MULTICLASS_H___
#define _LIBSVM_MULTICLASS_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/multiclass/MulticlassSVM.h>
#include <shogun/lib/external/shogun_libsvm.h>
#include <shogun/classifier/svm/LibSVM.h>

namespace shogun
{
/** @brief class LibSVMMultiClass. Does one vs one
 * classification. */
class MulticlassLibSVM : public MulticlassSVM
{
	public:
		/** default constructor */
		MulticlassLibSVM(LIBSVM_SOLVER_TYPE st=LIBSVM_C_SVC);

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		MulticlassLibSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab);

		/** destructor */
		virtual ~MulticlassLibSVM();

		/** get classifier type
		 *
		 * @return classifier type LIBSVMMULTICLASS
		 */
		virtual EMachineType get_classifier_type() { return CT_LIBSVMMULTICLASS; }

		/** @return object name */
		virtual const char* get_name() const { return "MulticlassLibSVM"; }

	protected:
		/** train multiclass SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(std::shared_ptr<Features> data=NULL);

	private:
		void register_params();

	protected:
		/** solver type */
		LIBSVM_SOLVER_TYPE solver_type;
};
}
#endif

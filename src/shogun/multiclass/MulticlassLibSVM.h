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
		MulticlassLibSVM(float64_t C, std::shared_ptr<Kernel> k );

		/** destructor */
		~MulticlassLibSVM() override;

		/** get classifier type
		 *
		 * @return classifier type LIBSVMMULTICLASS
		 */
		EMachineType get_classifier_type() override { return CT_LIBSVMMULTICLASS; }

		/** @return object name */
		const char* get_name() const override { return "MulticlassLibSVM"; }

	protected:
		/** train multiclass SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		bool train_machine(const std::shared_ptr<Features>&, const std::shared_ptr<Labels>& labs) override;

	private:
		void register_params();

	protected:
		/** solver type */
		LIBSVM_SOLVER_TYPE solver_type;
};
}
#endif

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, 
 *          Leon Kuchenbecker
 */

#ifndef _LIBSVM_H___
#define _LIBSVM_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/external/shogun_libsvm.h>

namespace shogun
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum LIBSVM_SOLVER_TYPE
{
	LIBSVM_C_SVC = 1,
	LIBSVM_NU_SVC = 2
};
#endif
/** @brief LibSVM */
class LibSVM : public SVM
{
	public:
		/** Default constructor, create a C-SVC svm */
		LibSVM();

		/** Constructor
		 *
		 * @param st solver type C or NU SVC
		 */
		LibSVM(LIBSVM_SOLVER_TYPE st);

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 * @param st solver type to use, C-SVC or nu-SVC
		 */
		LibSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab,
				LIBSVM_SOLVER_TYPE st=LIBSVM_C_SVC);

		~LibSVM() override;

		/** get classifier type
		 *
		 * @return classifier type LIBSVM
		 */
		EMachineType get_classifier_type() override { return CT_LIBSVM; }

		/** @return object name */
		const char* get_name() const override { return "LibSVM"; }

	private:
		void register_params();

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

	protected:
		/** solver type */
		LIBSVM_SOLVER_TYPE solver_type;
};
}
#endif

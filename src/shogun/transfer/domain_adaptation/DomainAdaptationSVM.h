/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Yuyu Zhang, Saurabh Goyal
 */

#ifdef USE_SVMLIGHT

#ifndef _DomainAdaptation_SVM_H___
#define _DomainAdaptation_SVM_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVMLight.h>


namespace shogun
{
/** @brief class DomainAdaptationSVM */
class DomainAdaptationSVM : public SVMLight
{
	public:

		/** default constructor */
		DomainAdaptationSVM();

		/** constructor
		 *
		 * @param C cost constant C
		 * @param k kernel
		 * @param lab labels
		 * @param presvm trained SVM to regularize against
		 * @param B trade-off constant B
		 */
		DomainAdaptationSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab, std::shared_ptr<SVM> presvm, float64_t B);

		/** destructor */
		virtual ~DomainAdaptationSVM();

		/** init SVM
		 *
		 * @param presvm trained SVM to regularize against
		 * @param B trade-off constant B
		 * */
		void init(const std::shared_ptr<SVM>& presvm, float64_t B);

		/** get classifier type
		 *
		 * @return classifier type
		 */
		virtual EMachineType get_classifier_type() { return CT_DASVM; }

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual std::shared_ptr<BinaryLabels> apply_binary(std::shared_ptr<Features> data=NULL);

		/** returns SVM that is used as prior information
		 *
		 * @return presvm
		 */
		virtual std::shared_ptr<SVM> get_presvm();

		/** getter for regularization parameter B
		 *
		 * @return regularization parameter B
		 */
		virtual float64_t get_B();

		/** getter for train_factor
		 *
		 * @return train_factor
		 */
		virtual float64_t get_train_factor();

		/** setter for train_factor
		 *
		 */
		virtual void set_train_factor(float64_t factor);

		/** @return object name */
		virtual const char* get_name() const { return "DomainAdaptationSVM"; }

	protected:

		/** check sanity of presvm
		 *
		 * @return true if sane, throws SG_ERROR otherwise
		 */
		virtual bool is_presvm_sane();

		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(std::shared_ptr<Features> data=NULL);

	private:
		void init();

	protected:

		/** SVM to regularize against */
		std::shared_ptr<SVM> presvm;

		/** regularization parameter B */
		float64_t B;

		/** flag to switch off regularization in training */
		float64_t train_factor;
};
}
#endif //_DomainAdaptation_SVM_H___
#endif //USE_SVMLIGHT

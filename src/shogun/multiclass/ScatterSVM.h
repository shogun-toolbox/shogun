/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Saurabh Goyal, 
 *          Leon Kuchenbecker
 */

#ifndef _SCATTERSVM_H___
#define _SCATTERSVM_H___

#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/multiclass/MulticlassSVM.h>
#include <shogun/lib/external/shogun_libsvm.h>


namespace shogun
{
	/** scatter svm variant */
	enum SCATTER_TYPE
	{
		/// no bias w/ libsvm
		NO_BIAS_LIBSVM,
#ifdef USE_SVMLIGHT
		/// no bias w/ svmlight
		NO_BIAS_SVMLIGHT,
#endif //USE_SVMLIGHT
		/// training with bias using test rule 1
		TEST_RULE1,
		/// training with bias using test rule 2
		TEST_RULE2
	};

/** @brief ScatterSVM - Multiclass SVM
 *
 * The ScatterSVM is an unpublished experimental
 * true multiclass SVM. Details are availabe
 * in the following technical report.
 *
 * This code is currently experimental.
 *
 * Robert Jenssen and Marius Kloft and Alexander Zien and S\"oren Sonnenburg and
 *           Klaus-Robert M\"{u}ller,
 * A Multi-Class Support Vector Machine Based on Scatter Criteria, TR 014-2009
 * TU Berlin, 2009
 *
 * */
class ScatterSVM : public MulticlassSVM
{
	public:
		/** default constructor  */
		ScatterSVM();

		/** constructor */
		ScatterSVM(SCATTER_TYPE type);

		/** constructor (using NO_BIAS as default scatter_type)
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		ScatterSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab);

		/** default destructor */
		virtual ~ScatterSVM();

		/** get classifier type
		 *
		 * @return classifier type LIBSVM
		 */
		virtual EMachineType get_classifier_type() { return CT_SCATTERSVM; }

		/** classify one example
		 *
		 * @param num number of example to classify
		 * @return resulting classification
		 */
		virtual float64_t apply_one(int32_t num);

		/** classify one vs rest
		 *
		 * @return resulting labels
		 */
		virtual std::shared_ptr<Labels> classify_one_vs_rest();

		/** @return object name */
		virtual const char* get_name() const { return "ScatterSVM"; }

	protected:
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
		void compute_norm_wc();
		virtual bool train_no_bias_libsvm();
#ifdef USE_SVMLIGHT
		virtual bool train_no_bias_svmlight();
#endif //USE_SVMLIGHT
		virtual bool train_testrule12();

		void register_params();

	protected:
		/** type of scatter SVM */
		SCATTER_TYPE scatter_type;

		/** norm of w_c */
		float64_t* norm_wc;
		int32_t norm_wc_len;

		/** norm of w_cw */
		float64_t* norm_wcw;
		int32_t norm_wcw_len;

		/** ScatterSVM rho */
		float64_t rho;

		/** number of classes */
		int32_t m_num_classes;
};
}
#endif // ScatterSVM

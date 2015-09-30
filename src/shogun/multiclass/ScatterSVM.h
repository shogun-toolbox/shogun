/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Written (W) 2009 Marius Kloft
 * Copyright (C) 2009 TU Berlin and Max-Planck-Society
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
class CScatterSVM : public CMulticlassSVM
{
	public:
		/** default constructor  */
		CScatterSVM();

		/** constructor */
		CScatterSVM(SCATTER_TYPE type);

		/** constructor (using NO_BIAS as default scatter_type)
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CScatterSVM(float64_t C, CKernel* k, CLabels* lab);

		/** default destructor */
		virtual ~CScatterSVM();

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
		virtual CLabels* classify_one_vs_rest();

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
		virtual bool train_machine(CFeatures* data=NULL);

	private:
		void compute_norm_wc();
		virtual bool train_no_bias_libsvm();
#ifdef USE_SVMLIGHT
		virtual bool train_no_bias_svmlight();
#endif //USE_SVMLIGHT
		virtual bool train_testrule12();

	protected:
		/** type of scatter SVM */
		SCATTER_TYPE scatter_type;

		/** SVM problem */
		svm_problem problem;
		/** SVM param */
		svm_parameter param;

		/** SVM model */
		struct svm_model* model;

		/** norm of w_c */
		float64_t* norm_wc;

		/** norm of w_cw */
		float64_t* norm_wcw;

		/** ScatterSVM rho */
		float64_t rho;

		/** number of classes */
		int32_t m_num_classes;
};
}
#endif // ScatterSVM

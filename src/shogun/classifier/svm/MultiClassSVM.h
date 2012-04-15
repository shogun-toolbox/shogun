/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MULTICLASSSVM_H___
#define _MULTICLASSSVM_H___

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/machine/KernelMulticlassMachine.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum EMultiClassSVM
{
	ONE_VS_REST,
	ONE_VS_ONE,
};
#endif

class CSVM;

/** @brief class MultiClassSVM */
class CMultiClassSVM : public CKernelMulticlassMachine
{
	public:
		/** default constructor  */
		CMultiClassSVM();

		/** constructor
		 *
		 * @param strategy multiclass strategy
		 */
		CMultiClassSVM(EMulticlassStrategy strategy);

		/** constructor
		 *
		 * @param strategy multiclass strategy
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CMultiClassSVM(
			EMulticlassStrategy strategy, float64_t C, CKernel* k, CLabels* lab);
		virtual ~CMultiClassSVM();

		/** create multiclass SVM
		 *
		 * @param num_classes number of classes in SVM
		 * @return if creation was successful
		 */
		bool create_multiclass_svm(int32_t num_classes);

		/** set SVM
		 *
		 * @param num number to set
		 * @param svm SVM to set
		 * @return if setting was successful
		 */
		bool set_svm(int32_t num, CSVM* svm);

		/** get SVM
		 *
		 * @param num which SVM to get
		 * @return SVM at number num
		 */
		CSVM* get_svm(int32_t num)
		{
			ASSERT(m_svms && m_num_svms>0);
			ASSERT(num>=0 && num<m_num_svms);
			SG_REF(m_svms[num]);
			return m_svms[num];
		}

		/** get number of SVMs
		 *
		 * @return number of SVMs
		 */
		int32_t inline get_num_svms()
		{
			return m_num_svms;
		}

		/** cleanup SVM */
		void cleanup();

		/** classify all examples
		 *
		 * @return resulting labels
		 */
		virtual CLabels* apply();

		/** classify all examples
		 *
		 * @return resulting labels
		 */
		virtual CLabels* apply(CFeatures* data);

		/** classify one example
		 *
		 * @param num number of example to classify
		 * @return resulting classification
		 */
		virtual float64_t apply(int32_t num);

		/** classify one vs rest
		 *
		 * @return resulting labels
		 */
		virtual CLabels* classify_one_vs_rest();

		/** classify one example one vs rest
		 *
		 * @param num number of example of classify
		 * @return resulting classification
		 */
		virtual float64_t classify_example_one_vs_rest(int32_t num);

		/** classify one vs one
		 *
		 * @return resulting labels
		 */
		CLabels* classify_one_vs_one();

		/** classify one example one vs one
		 *
		 * @param num number of example of classify
		 * @return resulting classification
		 */
		float64_t classify_example_one_vs_one(int32_t num);

		/** load a Multiclass SVM from file
		 * @param svm_file the file handle
		 */
		bool load(FILE* svm_file);

		/** write a Multiclass SVM to a file
		 * @param svm_file the file handle
		 */
		bool save(FILE* svm_file);

		/** get the type of multiclass'ness
		 *
		 * @return multiclass type 1 vs one etc
		 */
		inline EMultiClassSVM get_multiclass_type() { return multiclass_type; }

		// proxy of SVM getters
		SGVector<float64_t> get_linear_term() { return svm_proto()->get_linear_term(); }
		float64_t get_tube_epsilon() { return svm_proto()->get_tube_epsilon(); }
		float64_t get_epsilon() { return svm_proto()->get_epsilon(); }
		float64_t get_nu() { return svm_proto()->get_nu(); }
		float64_t get_C1() { return svm_proto()->get_C1(); }
		float64_t get_C2() { return svm_proto()->get_C2(); }
		int32_t get_qpsize() { return svm_proto()->get_qpsize(); }
		bool get_shrinking_enabled() { return svm_proto()->get_shrinking_enabled(); }
		float64_t get_objective() { return svm_proto()->get_objective(); }

		bool get_bias_enabled() { return svm_proto()->get_bias_enabled(); }
		bool get_linadd_enabled() { return svm_proto()->get_linadd_enabled(); }
		bool get_batch_computation_enabled() { return svm_proto()->get_batch_computation_enabled(); }

		// proxy of SVM setters
		void set_defaults(int32_t num_sv=0) { svm_proto()->set_defaults(num_sv); }
		void set_linear_term(SGVector<float64_t> linear_term) { svm_proto()->set_linear_term(linear_term); }
		void set_C(float64_t c_neg, float64_t c_pos) { svm_proto()->set_C(c_neg, c_pos); }
		void set_epsilon(float64_t eps) { svm_proto()->set_epsilon(eps); }
		void set_nu(float64_t nue) { svm_proto()->set_nu(nue); }
		void set_tube_epsilon(float64_t eps) { svm_proto()->set_tube_epsilon(eps); }
		void set_qpsize(int32_t qps) { svm_proto()->set_qpsize(qps); }
		void set_shrinking_enabled(bool enable) { svm_proto()->set_shrinking_enabled(enable); }
		void set_objective(float64_t v) { svm_proto()->set_objective(v); }

		void set_bias_enabled(bool enable_bias) { svm_proto()->set_bias_enabled(enable_bias); }
		void set_linadd_enabled(bool enable) { svm_proto()->set_linadd_enabled(enable); }
		void set_batch_computation_enabled(bool enable) { svm_proto()->set_batch_computation_enabled(enable); }

	protected:
		CSVM *svm_proto()
		{
			return dynamic_cast<CSVM*>(m_machine);
		}
		SGVector<int32_t> &svm_svs()
		{
			return svm_proto()->m_svs;
		}

	private:
		void init();

	protected:
		/** type of MultiClassSVM */
		EMultiClassSVM multiclass_type;

		/** number of classes */
		int32_t m_num_classes;
		/** number of SVMs */
		int32_t m_num_svms;
		/** the SVMs */
		CSVM** m_svms;
};
}
#endif

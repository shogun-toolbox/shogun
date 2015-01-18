/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2011 Soeren Sonnenburg
 * Written (W) 2012 Soeren Sonnenburg, Chiyuan Zhang
 * Copyright (C) 1999-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MULTICLASSSVM_H___
#define _MULTICLASSSVM_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/machine/KernelMulticlassMachine.h>

namespace shogun
{

class CSVM;

/** @brief class MultiClassSVM */
class CMulticlassSVM : public CKernelMulticlassMachine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS);

		/** default constructor  */
		CMulticlassSVM();

		/** constructor
		 *
		 * @param strategy multiclass strategy
		 */
		CMulticlassSVM(CMulticlassStrategy *strategy);

		/** constructor
		 *
		 * @param strategy multiclass strategy
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CMulticlassSVM(
			CMulticlassStrategy *strategy, float64_t C, CKernel* k, CLabels* lab);
		virtual ~CMulticlassSVM();

		/** create multiclass SVM. Appends the appropriate number of svm pointer
		 * (depending on multiclass strategy) to m_machines. All pointers are
		 * initialized with NULL.
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
			return dynamic_cast<CSVM *>(m_machines->get_element_safe(num));
		}

		/** load a Multiclass SVM from file
		 * @param svm_file the file handle
		 */
		bool load(FILE* svm_file);

		/** write a Multiclass SVM to a file
		 * @param svm_file the file handle
		 */
		bool save(FILE* svm_file);

		// TODO remove if unnecessary here
		/** get linear term of base SVM
		 * @return linear term of base SVM
		 */
		SGVector<float64_t> get_linear_term() { return svm_proto()->get_linear_term(); }
		// TODO remove if unnecessary here
		/** get tube epsilon of base SVM
		 * @return tube epsilon of base SVM
		 */
		float64_t get_tube_epsilon() { return svm_proto()->get_tube_epsilon(); }
		// TODO remove if unnecessary here
		/** get epsilon of base SVM
		 * @return epsilon of base SVM
		 */
		float64_t get_epsilon() { return svm_proto()->get_epsilon(); }
		// TODO remove if unnecessary here
		/** get nu of base SVM
		 * @return nu of base SVM
		 */
		float64_t get_nu() { return svm_proto()->get_nu(); }
		// TODO remove if unnecessary here
		/** get C of base SVM
		 * @return C of base SVM
		 */
		float64_t get_C() { return m_C; }
		// TODO remove if unnecessary here
		/** get qpsize of base SVM
		 * @return qpsize of base SVM
		 */
		int32_t get_qpsize() { return svm_proto()->get_qpsize(); }
		// TODO remove if unnecessary here
		/** get shrinking option of base SVM
		 * @return whether shrinking of base SVM is enabled
		 */
		bool get_shrinking_enabled() { return svm_proto()->get_shrinking_enabled(); }
		// TODO remove if unnecessary here
		/** get objective of base SVM
		 * @return objective of base SVM
		 */
		float64_t get_objective() { return svm_proto()->get_objective(); }

		// TODO remove if unnecessary here
		/** get bias enabled options of base SVM
		 * @return whether bias of base SVM is enabled
		 */
		bool get_bias_enabled() { return svm_proto()->get_bias_enabled(); }
		// TODO remove if unnecessary here
		/** get linadd option of base SVM
		 * @return whether linadd of base SVM is enabled
		 */
		bool get_linadd_enabled() { return svm_proto()->get_linadd_enabled(); }
		// TODO remove if unnecessary here
		/** get batch computation option of base SVM
		 * @return whether batch computation of base SVM is enabled
		 */
		bool get_batch_computation_enabled() { return svm_proto()->get_batch_computation_enabled(); }

		// TODO remove if unnecessary here
		/** set default number of support vectors
		 * @param num_sv number of support vectors
		 */
		void set_defaults(int32_t num_sv=0) { svm_proto()->set_defaults(num_sv); }
		// TODO remove if unnecessary here
		/** set linear term
		 * @param linear_term linear term vector
		 */
		void set_linear_term(SGVector<float64_t> linear_term) { svm_proto()->set_linear_term(linear_term); }
		// TODO remove if unnecessary here
		/** set C parameters
		 * @param C set regularization parameter
		 */
		void set_C(float64_t C) { svm_proto()->set_C(C,C); m_C = C; }
		// TODO remove if unnecessary here
		/** set epsilon value
		 * @param eps epsilon value
		 */
		void set_epsilon(float64_t eps) { svm_proto()->set_epsilon(eps); }
		// TODO remove if unnecessary here
		/** set nu value
		 * @param nue nu value
		 */
		void set_nu(float64_t nue) { svm_proto()->set_nu(nue); }
		// TODO remove if unnecessary here
		/** set tube epsilon value
		 * @param eps tube epsilon value
		 */
		void set_tube_epsilon(float64_t eps) { svm_proto()->set_tube_epsilon(eps); }
		// TODO remove if unnecessary here
		/** set set QP size
		 * @param qps qp size
		 */
		void set_qpsize(int32_t qps) { svm_proto()->set_qpsize(qps); }
		// TODO remove if unnecessary here
		/** set shrinking option
		 * @param enable whether shrinking should be enabled
		 */
		void set_shrinking_enabled(bool enable) { svm_proto()->set_shrinking_enabled(enable); }
		// TODO remove if unnecessary here
		/** set objective value
		 * @param v objective value
		 */
		void set_objective(float64_t v) { svm_proto()->set_objective(v); }
		// TODO remove if unnecessary here
		/** set bias option
		 * @param enable_bias whether bias should be enabled
		 */
		void set_bias_enabled(bool enable_bias) { svm_proto()->set_bias_enabled(enable_bias); }
		// TODO remove if unnecessary here
		/** set linadd option
		 * @param enable whether linadd should be enabled
		 */
		void set_linadd_enabled(bool enable) { svm_proto()->set_linadd_enabled(enable); }
		// TODO remove if unnecessary here
		/** set batch computation option
		 * @param enable whether batch computation should be enabled
		 */
		void set_batch_computation_enabled(bool enable) { svm_proto()->set_batch_computation_enabled(enable); }

		/** @return name of SGSerializable */
		virtual const char* get_name() const
		{
			return "MulticlassSVM";
		}

	protected:

		/** casts m_machine to SVM */
		CSVM *svm_proto()
		{
			return dynamic_cast<CSVM*>(m_machine);
		}
		/** returns support vectors */
		SGVector<int32_t> svm_svs()
		{
			return svm_proto()->m_svs;
		}

		/** initializes machines (OvO, OvR) for apply */
		virtual bool init_machines_for_apply(CFeatures* data);

		/** is machine an SVM instance */
		virtual bool is_acceptable_machine(CMachine *machine)
		{
			CSVM *svm = dynamic_cast<CSVM*>(machine);
			if (svm == NULL)
				return false;
			return true;
		}

	private:

		void init();

	protected:

		/** C regularization constant */
		float64_t m_C;
};
}
#endif

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SVM_H___
#define _SVM_H___

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/machine/KernelMachine.h>

namespace shogun
{

class CMKL;
class CMulticlassSVM;

/** @brief A generic Support Vector Machine Interface.
 *
 * A support vector machine is defined as
 *  \f[
 *		f({\bf x})=\sum_{i=0}^{N-1} \alpha_i k({\bf x}, {\bf x_i})+b
 *	\f]
 *
 * where \f$N\f$ is the number of training examples
 * \f$\alpha_i\f$ are the weights assigned to each training example
 * \f$k(x,x')\f$ is the kernel
 * and \f$b\f$ the bias.
 *
 * Using an a-priori choosen kernel, the \f$\alpha_i\f$ and bias are determined
 * by solving the following quadratic program
 *
 * \f{eqnarray*}
 *		\max_{\bf \alpha} && \sum_{i=0}^{N-1} \alpha_i - \sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \alpha_i y_i \alpha_j y_j  k({\bf x_i}, {\bf x_j})\\
 *		\mbox{s.t.} && 0\leq\alpha_i\leq C\\
 *					&& \sum_{i=0}^{N-1} \alpha_i y_i=0\\
 * \f}
 * here C is a pre-specified regularization parameter.
 */
class CSVM : public CKernelMachine
{
	public:

		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** Create an empty Support Vector Machine Object
		 * @param num_sv with num_sv support vectors
		 */
		CSVM(int32_t num_sv=0);

		/** Create a Support Vector Machine Object from a
		 * trained SVM
		 *
		 * @param C the C parameter
		 * @param k the Kernel object
		 * @param lab the Label object
		 */
		CSVM(float64_t C, CKernel* k, CLabels* lab);

		virtual ~CSVM();

		/** set default values for members a SVM object
		*/
		void set_defaults(int32_t num_sv=0);


		/**
		 * get linear term
		 *
		 * @return the linear term
		 */
		virtual SGVector<float64_t> get_linear_term();


		/**
		 * set linear term of the QP
		 *
		 * @param linear_term the linear term
		 */
		virtual void set_linear_term(const SGVector<float64_t> linear_term);


		/** load a SVM from file
		 * @param svm_file the file handle
		 */
		bool load(FILE* svm_file);

		/** write a SVM to a file
		 * @param svm_file the file handle
		 */
		bool save(FILE* svm_file);

		/** set nu
		 *
		 * @param nue new nu
		 */
		inline void set_nu(float64_t nue) { nu=nue; }


		/** set C
		 *
		 * @param c_neg new C constant for negatively labeled examples
		 * @param c_pos new C constant for positively labeled examples
		 *
		 * Note that not all SVMs support this (however at least CLibSVM and
		 * CSVMLight do)
		 */
		inline void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }


		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** set tube epsilon
		 *
		 * @param eps new tube epsilon
		 */
		inline void set_tube_epsilon(float64_t eps) { tube_epsilon=eps; }

		/** get tube epsilon
		 *
		 * @return tube epsilon
		 */
		inline float64_t get_tube_epsilon() { return tube_epsilon; }

		/** set qpsize
		 *
		 * @param qps new qpsize
		 */
		inline void set_qpsize(int32_t qps) { qpsize=qps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/** get nu
		 *
		 * @return nu
		 */
		inline float64_t get_nu() { return nu; }

		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline float64_t get_C2() { return C2; }

		/** get qpsize
		 *
		 * @return qpsize
		 */
		inline int32_t get_qpsize() { return qpsize; }

		/** set state of shrinking
		 *
		 * @param enable if shrinking will be enabled
		 */
		inline void set_shrinking_enabled(bool enable)
		{
			use_shrinking=enable;
		}

		/** get state of shrinking
		 *
		 * @return if shrinking is enabled
		 */
		inline bool get_shrinking_enabled()
		{
			return use_shrinking;
		}

		/** compute svm dual objective
		 *
		 * @return computed dual objective
		 */
		float64_t compute_svm_dual_objective();

		/** compute svm primal objective
		 *
		 * @return computed svm primal objective
		 */
		float64_t compute_svm_primal_objective();

		/** set objective
		 *
		 * @param v objective
		 */
		inline void set_objective(float64_t v)
		{
			objective=v;
		}

		/** get objective
		 *
		 * @return objective
		 */
		inline float64_t get_objective()
		{
			return objective;
		}

		/** set callback function svm optimizers may call when they have a new
		 * (small) set of alphas
		 *
		 * @param m pointer to mkl object
		 * @param cb callback function
		 *
		 * */
		void set_callback_function(CMKL* m, bool (*cb)
				(CMKL* mkl, const float64_t* sumw, const float64_t suma));

		/** @return object name */
		virtual const char* get_name() const { return "SVM"; }

	protected:

		/**
		 * get linear term copy as dynamic array
		 *
		 * @return linear term copied to a dynamic array
		 */
		virtual float64_t* get_linear_term_array();

		/** linear term in qp */
		SGVector<float64_t> m_linear_term;

		/** if SVM is loaded */
		bool svm_loaded;
		/** epsilon */
		float64_t epsilon;
		/** tube epsilon for support vector regression*/
		float64_t tube_epsilon;
		/** nu */
		float64_t nu;
		/** C1 regularization const*/
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** objective */
		float64_t objective;
		/** qpsize */
		int32_t qpsize;
		/** if shrinking shall be used */
		bool use_shrinking;

		/** callback function svm optimizers may call when they have a new
		 * (small) set of alphas */
		bool (*callback) (CMKL* mkl, const float64_t* sumw, const float64_t suma);
		/** mkl object that svm optimizers need to pass when calling the callback
		 * function */
		CMKL* mkl;

	friend class CMulticlassSVM;
};
}
#endif

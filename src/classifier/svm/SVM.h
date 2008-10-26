/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SVM_H___
#define _SVM_H___

#include "lib/common.h"
#include "features/Features.h"
#include "kernel/Kernel.h"
#include "kernel/KernelMachine.h"

class CKernelMachine;

/** A generic Support Vector Machine Interface
 * 
 * A support vector machine is defined as
 *  \f[
 * 		f({\bf x})=\sum_{i=0}^{N-1} \alpha_i k({\bf x}, {\bf x_i})+b
 * 	\f]
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
 * 		\max_{\bf \alpha} && \sum_{i=0}^{N-1} \alpha_i - \sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \alpha_i y_i \alpha_j y_j  k({\bf x_i}, {\bf x_j})\\
 * 		\mbox{s.t.} && 0\leq\alpha_i\leq C\\
 * 		 			&& \sum_{i=0}^{N-1} \alpha_i y_i=0\\
 * \f}
 * here C is a pre-specified regularization parameter.
 */
class CSVM : public CKernelMachine
{
	public:
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
		CSVM(DREAL C, CKernel* k, CLabels* lab);
		virtual ~CSVM();

		/** set default values for members a SVM object
		*/
		void set_defaults(int32_t num_sv=0);

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
		inline void set_nu(DREAL nue) { nu=nue; }

		/** set C
		 *
		 * @param c1 new C constant for negatively labelled examples
		 * @param c2 new C constant for positively labelled examples
		 *
		 * Note that not all SVMs support this (however at least CLibSVM and
		 * CSVMLight do)
		 */
		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		/** set epsilon for weights
		 *
		 * @param eps new weight_epsilon
		 */
		inline void set_weight_epsilon(DREAL eps) { weight_epsilon=eps; }

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(DREAL eps) { epsilon=eps; }

		/** set tube epsilon
		 *
		 * @param eps new tube epsilon
		 */
		inline void set_tube_epsilon(DREAL eps) { tube_epsilon=eps; }

		/** set C mkl
		 *
		 * @param C new C_mkl
		 */
		inline void set_C_mkl(DREAL C) { C_mkl = C; }

		/** set mkl norm
		 *
		 * @param norm new mkl norm (1 or 2)
		 */
		inline void set_mkl_norm(int32_t norm)
		{
			if (norm!=1 && norm!=2)
				SG_ERROR("Only 1-and 2-norm supported\n");
			mkl_norm = norm;
		}

		/** set qpsize
		 *
		 * @param qps new qpsize
		 */
		inline void set_qpsize(int32_t qps) { qpsize=qps; }

		/** set state of bias
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** get state of bias
		 *
		 * @return state of bias
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** get epsilon for weights
		 *
		 * @return epsilon for weights
		 */
		inline DREAL get_weight_epsilon() { return weight_epsilon; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline DREAL get_epsilon() { return epsilon; }

		/** get nu
		 *
		 * @return nu
		 */
		inline DREAL get_nu() { return nu; }

		/** get C1
		 *
		 * @return C1
		 */
		inline DREAL get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline DREAL get_C2() { return C2; }

		/** get qpsize
		 *
		 * @return qpsize
		 */
		inline int32_t get_qpsize() { return qpsize; }

		/** get support vector at given index
		 *
		 * @param idx index of support vector
		 * @return support vector
		 */
		inline int32_t get_support_vector(int32_t idx)
		{
			ASSERT(svm_model.svs && idx<svm_model.num_svs);
			return svm_model.svs[idx];
		}

		/** get alpha at given index
		 *
		 * @param idx index of alpha
		 * @return alpha
		 */
		inline DREAL get_alpha(int32_t idx)
		{
			ASSERT(svm_model.alpha && idx<svm_model.num_svs);
			return svm_model.alpha[idx];
		}

		/** set support vector at given index to given value
		 *
		 * @param idx index of support vector
		 * @param val new value of support vector
		 * @return if operation was successful
		 */
		inline bool set_support_vector(int32_t idx, int32_t val)
		{
			if (svm_model.svs && idx<svm_model.num_svs)
				svm_model.svs[idx]=val;
			else
				return false;

			return true;
		}

		/** set alpha at given index to given value
		 *
		 * @param idx index of alpha vector
		 * @param val new value of alpha vector
		 * @return if operation was successful
		 */
		inline bool set_alpha(int32_t idx, DREAL val)
		{
			if (svm_model.alpha && idx<svm_model.num_svs)
				svm_model.alpha[idx]=val;
			else
				return false;

			return true;
		}

		/** get bias
		 *
		 * @return bias
		 */
		inline DREAL get_bias()
		{
			return svm_model.b;
		}

		/** set bias to given value
		 *
		 * @param bias new bias
		 */
		inline void set_bias(DREAL bias)
		{
			svm_model.b=bias;
		}

		/** get number of support vectors
		 *
		 * @return number of support vectors
		 */
		inline int32_t get_num_support_vectors()
		{
			return svm_model.num_svs;
		}

		/** set alphas to given values
		 *
		 * @param alphas array with all alphas to set
		 * @param d number of alphas (== number of support vectors)
		 */
		void set_alphas(DREAL* alphas, int32_t d)
		{
			ASSERT(alphas);
			ASSERT(d==svm_model.num_svs);

			for(int32_t i=0; i<d; i++)
				svm_model.alpha[i]=alphas[i];
		}

		/** set support vectors to given values
		 *
		 * @param svs array with all support vectors to set
		 * @param d number of support vectors
		 */
		void set_support_vectors(int32_t* svs, int32_t d)
		{
			ASSERT(svs);
			ASSERT(d==svm_model.num_svs);

			for(int32_t i=0; i<d; i++)
				svm_model.svs[i]=svs[i];
		}

		/** get all support vectors (swig compatible)
		 *
		 * @param svs array to contain a copy of the support vectors
		 * @param num number of support vectors in the array
		 */
		void get_support_vectors(int32_t** svs, int32_t* num)
		{
			int32_t nsv = get_num_support_vectors();

			ASSERT(svs && num);
			*svs=NULL;
			*num=nsv;

			if (nsv>0)
			{
				*svs = (int32_t*) malloc(sizeof(int32_t)*nsv);
				for(int32_t i=0; i<nsv; i++)
					(*svs)[i] = get_support_vector(i);
			}
		}

		/** get all alphas (swig compatible)
		 *
		 * @param alphas array to contain a copy of the alphas
		 * @param d1 number of alphas in the array
		 */
		void get_alphas(DREAL** alphas, int32_t* d1)
		{
			int32_t nsv = get_num_support_vectors();

			ASSERT(alphas && d1);
			*alphas=NULL;
			*d1=nsv;

			if (nsv>0)
			{
				*alphas = (DREAL*) malloc(nsv*sizeof(DREAL));
				for(int32_t i=0; i<nsv; i++)
					(*alphas)[i] = get_alpha(i);
			}
		}

		/** create new model
		 *
		 * @param num number of alphas and support vectors in new model
		 */
		inline bool create_new_model(int32_t num)
		{
			delete[] svm_model.alpha;
			delete[] svm_model.svs;

			svm_model.b=0;
			svm_model.num_svs=num;

			if (num>0)
			{
				svm_model.alpha= new double[num];
				svm_model.svs= new int32_t[num];
				return (svm_model.alpha!=NULL && svm_model.svs!=NULL);
			}
			else
			{
				svm_model.alpha= NULL;
				svm_model.svs=NULL;
				return true;
			}
		}

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

		/** set state of mkl
		 *
		 * @param enable if mkl shall be enabled
		 */
		inline void set_mkl_enabled(bool enable)
		{
			use_mkl=enable;
		}

		/** get state of mkl
		 *
		 * @return if mkl is enabled
		 */
		inline bool get_mkl_enabled()
		{
			return use_mkl;
		}

		/** compute objective
		 *
		 * @return computed objective
		 */
		DREAL compute_objective();

		/** set objective
		 *
		 * @param v objective
		 */
		inline void set_objective(DREAL v)
		{
			objective=v;
		}

		/** get objective
		 *
		 * @return objective
		 */
		inline DREAL get_objective()
		{
			return objective;
		}

		/** initialise kernel optimisation
		 *
		 * @return if operation was successful
		 */
		bool init_kernel_optimization();

		/** classify SVM
		 *
		 * @param lab classified labels
		 * @return classified labels
		 */
		virtual CLabels* classify(CLabels* lab=NULL);

		/** classify one example
		 *
		 * @param num which example to classify
		 * @return classified value
		 */
		virtual DREAL classify_example(int32_t num);

		/** classify example helper, used in threads
		 *
		 * @param p params of the thread
		 * @return nothing really
		 */
		static void* classify_example_helper(void* p);

	protected:
		/// an SVM is defined by support vectors, their coefficients alpha
		/// and the bias b ( + CKernelMachine::kernel)
		struct TModel
		{
			/** bias b */
			DREAL b;
			/** array of coefficients alpha */
			DREAL* alpha;
			/** array of support vectors */
			int32_t* svs;
			/** number of support vectors */
			int32_t num_svs;
		};

		/** SVM's model */
		TModel svm_model;
		/** if SVM is loaded */
		bool svm_loaded;
		/** epsilon for multiple kernel learning */
		DREAL weight_epsilon;
		/** epsilon */
		DREAL epsilon;
		/** tube epsilon for support vector regression*/
		DREAL tube_epsilon;
		/** nu */
		DREAL nu;
		/** C1 regularization const*/
		DREAL C1;
		/** C2 */
		DREAL C2;
		/** norm used in mkl, can be 1 or 2 */
		int32_t  mkl_norm;
		/** C_mkl */
		DREAL C_mkl;
		/** objective */
		DREAL objective;
		/** qpsize */
		int32_t qpsize;
		/** if bias shall be used */
		bool use_bias;
		/** if shrinking shall be used */
		bool use_shrinking;
		/** if mkl shall be used */
		bool use_mkl;
};
#endif

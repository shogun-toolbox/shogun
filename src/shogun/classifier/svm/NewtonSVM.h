/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Harshit Syal
 * Copyright (C) 2012 Harshit Syal
 */

#ifndef _NEWTONSVM_H___
#define _NEWTONSVM_H___

#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Labels.h>

namespace shogun
{
/** @brief class NewtonSVM */
class CNewtonSVM : public CLinearMachine
{
	public:
		/** default constructor */
		CNewtonSVM();

		/** constructor
		 * @param C constant C
	         * @param itr constant no of iterations
		 * @param traindat training features
		 * @param trainlab labels for features
		 */
		CNewtonSVM(float64_t C, CDotFeatures* traindat, CLabels* trainlab,int32_t itr=20);
		
		virtual ~CNewtonSVM();

		/** get classifier type
		 *
		 * @return classifier type NewtonSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_NEWTONSVM; }

		/**
		 * set C
	         * @param C constant C
		 */
		inline void set_C(float64_t c) { C=c; }

		/** get epsilon
		 *  @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/**
		 * set epsilon
	         * @param epsilon constant epsilon
		 */
		inline void set_epsilon(float64_t e) { epsilon=e; }

		/** get C
		 *  @return C
		 */
		inline float64_t get_C() { return C; }

		
		/** set if bias shall be enabled
		 *  @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** get if bias is enabled
		 *  @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** set num_iter
		 *  @return num_iter 
		 */
		inline int32_t get_num_iter() {return num_iter;}

		/** set iter
		 *  @param num_iter number of iterations
		 */
		inline void set_num_iter(int32_t iter) { num_iter=iter; }
		
		/** @return object name */
		inline virtual const char* get_name() const { return "NewtonSVM"; }
	
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

		void obj_fun_linear(float64_t* weights,float64_t* out,float64_t* obj,int32_t* sv,int32_t* numsv,float64_t* grad);

		void line_search_linear(float64_t* weights,float64_t* d,float64_t* out,float64_t* tx);

	protected:
		/** lambda=1/C */
		float64_t lambda,C,epsilon;
		float64_t prec;
		int32_t x_n,x_d,num_iter;
		/** if bias is used */
		bool use_bias;
};
}
#endif

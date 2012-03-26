/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Harshit Syal
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
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
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab labels for features
		 */
		CNewtonSVM(float64_t l,int32_t itr, CDotFeatures* traindat, CLabels* trainlab);
		virtual ~CNewtonSVM();

		/** get classifier type
		 *
		 * @return classifier type NewtonSVM
		 */
		//virtual inline EClassifierType get_classifier_type() { return CT_NEWTONSVM; }

		/** set C
		 *
		 * @param c_neg new C constant for negatively labeled examples
		 * @param c_pos new C constant for positively labeled examples
		 *
		 */
		inline void set_lambda(float64_t l) { lambda=l; }

		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_lambda() { return lambda; }

		/** get C2
		 *
		 * @return C2
		 */

		/** set if bias shall be enabled
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** get if bias is enabled
		 *
		 * @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline int32_t get_num_iter() {return num_iter;}
		inline void set_num_iter(int32_t iter) { num_iter=iter; }
		
		inline float64_t get_epsilon() { return epsilon; }

		/** @return object name */
		inline virtual const char* get_name() const { return "NewtonSVM"; }

		inline void createDiagnolMatrix(float64_t *matrix,float64_t *v,int32_t size)
		{		
			for(int32_t i=0;i<size;i++)
			for(int32_t j=0;j<size;j++)
			{
				if(i==j)
				matrix[j*size+i]=v[i];
				else
				matrix[j*size+i]=0;
		}
	
}

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
		void obj_fun_linear(float64_t *w,CDotFeatures *features,float64_t *out,float64_t *obj,int32_t *sv,int32_t *numsv,float64_t *grad,SGVector<float64_t> v);
		void line_search_linear(float64_t *w,float64_t *d,float64_t *out,SGVector<float64_t> Y,float64_t lambda,float64_t *tx);
	protected:
		/** C1 */
		float64_t lambda;
		/** epsilon */
		float64_t epsilon,prec;
		int32_t x_n,x_d,num_iter;
		/** if bias is used */
		bool use_bias;
};
}
#endif

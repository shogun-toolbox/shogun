/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNEL_MACHINE_H__
#define _KERNEL_MACHINE_H__

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/features/Labels.h>
#include <shogun/machine/Machine.h>

#include <stdio.h>

namespace shogun
{
class CMachine;
class CLabels;
class CKernel;

/** @brief A generic KernelMachine interface.
 *
 * A kernel machine is defined as
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
 * in a training procedure.
 */
class CKernelMachine : public CMachine
{
	public:
		/** default constructor */
		CKernelMachine();

		/** destructor */
		virtual ~CKernelMachine();

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name(void) const {
			return "KernelMachine"; }


		virtual bool train(CFeatures* data=NULL)
		{
			bool result=train_kernel_machine(data);

			if (m_store_sv_features)
				store_sv_features();

			return result;
		}

		/** set kernel
		 *
		 * @param k kernel
		 */
		inline void set_kernel(CKernel* k)
		{
			SG_UNREF(kernel);
			SG_REF(k);
			kernel=k;
		}

		/** get kernel
		 *
		 * @return kernel
		 */
		inline CKernel* get_kernel()
		{
			SG_REF(kernel);
			return kernel;
		}

		/** set batch computation enabled
		 *
		 * @param enable if batch computation shall be enabled
		 */
		inline void set_batch_computation_enabled(bool enable)
		{
			use_batch_computation=enable;
		}

		/** check if batch computation is enabled
		 *
		 * @return if batch computation is enabled
		 */
		inline bool get_batch_computation_enabled()
		{
			return use_batch_computation;
		}

		/** set linadd enabled
		 *
		 * @param enable if linadd shall be enabled
		 */
		inline void set_linadd_enabled(bool enable)
		{
			use_linadd=enable;
		}

		/** check if linadd is enabled
		 *
		 * @return if linadd is enabled
		 */
		inline bool get_linadd_enabled()
		{
			return use_linadd ;
		}

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

		/** get bias
		 *
		 * @return bias
		 */
		inline float64_t get_bias()
		{
			return m_bias;
		}

		/** set bias to given value
		 *
		 * @param bias new bias
		 */
		inline void set_bias(float64_t bias)
		{
			m_bias=bias;
		}

		/** get support vector at given index
		 *
		 * @param idx index of support vector
		 * @return support vector
		 */
		inline int32_t get_support_vector(int32_t idx)
		{
			ASSERT(m_svs.vector && idx<m_svs.vlen);
			return m_svs.vector[idx];
		}

		/** get alpha at given index
		 *
		 * @param idx index of alpha
		 * @return alpha
		 */
		inline float64_t get_alpha(int32_t idx)
		{
			ASSERT(m_alpha.vector && idx<m_svs.vlen);
			return m_alpha.vector[idx];
		}

		/** set support vector at given index to given value
		 *
		 * @param idx index of support vector
		 * @param val new value of support vector
		 * @return if operation was successful
		 */
		inline bool set_support_vector(int32_t idx, int32_t val)
		{
			if (m_svs.vector && idx<m_svs.vlen)
				m_svs.vector[idx]=val;
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
		inline bool set_alpha(int32_t idx, float64_t val)
		{
			if (m_alpha.vector && idx<m_svs.vlen)
				m_alpha.vector[idx]=val;
			else
				return false;

			return true;
		}

		/** get number of support vectors
		 *
		 * @return number of support vectors
		 */
		inline int32_t get_num_support_vectors()
		{
			return m_svs.vlen;
		}

		/** set alphas to given values
		 *
		 * @param alphas array with all alphas to set
		 * @param d number of alphas (== number of support vectors)
		 */
		void set_alphas(float64_t* alphas, int32_t d)
		{
			ASSERT(alphas);
			ASSERT(m_alpha.vector);
			ASSERT(d==m_svs.vlen);

			for(int32_t i=0; i<d; i++)
				m_alpha.vector[i]=alphas[i];
		}

		/** set support vectors to given values
		 *
		 * @param svs array with all support vectors to set
		 * @param d number of support vectors
		 */
		void set_support_vectors(int32_t* svs, int32_t d)
		{
			ASSERT(m_svs.vector);
			ASSERT(svs);
			ASSERT(d==m_svs.vlen);

			for(int32_t i=0; i<d; i++)
				m_svs.vector[i]=svs[i];
		}

		/** get all support vectors
		 *
		 */
		SGVector<int32_t> get_support_vectors()
		{
			int32_t nsv = get_num_support_vectors();
			int32_t* svs = NULL;

			if (nsv>0)
			{
				svs = (int32_t*) SG_MALLOC(sizeof(int32_t)*nsv);
				for(int32_t i=0; i<nsv; i++)
					svs[i] = get_support_vector(i);
			}

			return SGVector<int32_t>(svs,nsv);
		}

		/** get all alphas
		 *
		 */
		SGVector<float64_t> get_alphas()
		{
			int32_t nsv = get_num_support_vectors();
			float64_t* alphas = NULL;

			if (nsv>0)
			{
				alphas = (float64_t*) SG_MALLOC(nsv*sizeof(float64_t));
				for(int32_t i=0; i<nsv; i++)
					alphas[i] = get_alpha(i);
			}

			return SGVector<float64_t>(alphas,nsv);
		}

		/** create new model
		 *
		 * @param num number of alphas and support vectors in new model
		 */
		inline bool create_new_model(int32_t num)
		{
			delete[] m_alpha.vector;
			delete[] m_svs.vector;

			m_bias=0;
			m_svs.vlen=num;

			if (num>0)
			{
				m_alpha.vector= new float64_t[num];
				m_svs.vector= new int32_t[num];
				return (m_alpha.vector!=NULL && m_svs.vector!=NULL);
			}
			else
			{
				m_alpha.vector= NULL;
				m_svs.vector=NULL;
				return true;
			}
		}

		/** initialise kernel optimisation
		 *
		 * @return if operation was successful
		 */
		bool init_kernel_optimization();

		/** apply kernel machine to all objects
		 *
		 * @return result labels
		 */
		virtual CLabels* apply();

		/** apply kernel machine to data
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data);

		/** apply kernel machine to one example
		 *
		 * @param num which example to apply to
		 * @return classified value
		 */
		virtual float64_t apply(int32_t num);

		/** apply example helper, used in threads
		 *
		 * @param p params of the thread
		 * @return nothing really
		 */
		static void* apply_helper(void* p);

		/** TODO comment
		 *
		 */
		inline void set_store_sv_features(bool store)
		{
			m_store_sv_features=store;
		}

	protected:
		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @return whether training was successful
		 */
		virtual bool train_kernel_machine(CFeatures* data=NULL)
		{
			SG_NOTIMPLEMENTED;
			return false;
		}

		virtual void store_sv_features();

	protected:
		/** kernel */
		CKernel* kernel;
		/** if batch computation is enabled */
		bool use_batch_computation;
		/** if linadd is enabled */
		bool use_linadd;
		/** if bias shall be used */
		bool use_bias;
		/**  bias term b */
		float64_t m_bias;

		/** coefficients alpha */
		SGVector<float64_t> m_alpha;

		/** array of ``support vectors'' (indices of feature objects) */
		SGVector<int32_t> m_svs;

		bool m_store_sv_features;
};
}
#endif /* _KERNEL_MACHINE_H__ */

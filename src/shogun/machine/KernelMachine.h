/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNEL_MACHINE_H__
#define _KERNEL_MACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>

#include <stdio.h>

namespace shogun
{
class CLabels;
class CBinaryLabels;
class CRegressionLabels;
class CKernel;
class CCustomKernel;
class CFeatures;

/** @brief A generic KernelMachine interface.
 *
 * A kernel machine is defined as
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
 * in a training procedure.
 *
 * Kernel machines support locking. This means that given some data, the machine
 * can be locked. When this is done, the kernel matrix is computed and stored in
 * a custom kernel. Speeds up cross-validation. Only train_locked and
 * apply_locked are available when locked.
 */
class CKernelMachine : public CMachine
{
	public:
		/** default constructor */
		CKernelMachine();

		/** Convenience constructor to initialize a trained kernel
		 * machine
		 *
		 * @param k kernel
		 * @param alphas vector of alpha weights
		 * @param svs indices of examples, i.e. i's for x_i
		 * @param b bias term
		 */
		CKernelMachine(CKernel* k, const SGVector<float64_t> alphas, const SGVector<int32_t> svs, float64_t b);

		/** copy constructor
		 * @param machine machine having parameters to copy
		 */
		CKernelMachine(CKernelMachine* machine);

		/** destructor */
		virtual ~CKernelMachine();

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "KernelMachine"; }

		/** set kernel
		 *
		 * @param k kernel
		 */
		void set_kernel(CKernel* k);

		/** get kernel
		 *
		 * @return kernel
		 */
		CKernel* get_kernel();

		/** set batch computation enabled
		 *
		 * @param enable if batch computation shall be enabled
		 */
		void set_batch_computation_enabled(bool enable);

		/** check if batch computation is enabled
		 *
		 * @return if batch computation is enabled
		 */
		bool get_batch_computation_enabled();

		/** set linadd enabled
		 *
		 * @param enable if linadd shall be enabled
		 */
		void set_linadd_enabled(bool enable);

		/** check if linadd is enabled
		 *
		 * @return if linadd is enabled
		 */
		bool get_linadd_enabled();

		/** set state of bias
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		void set_bias_enabled(bool enable_bias);

		/** get state of bias
		 *
		 * @return state of bias
		 */
		bool get_bias_enabled();

		/** get bias
		 *
		 * @return bias
		 */
		float64_t get_bias();

		/** set bias to given value
		 *
		 * @param bias new bias
		 */
		void set_bias(float64_t bias);

		/** get support vector at given index
		 *
		 * @param idx index of support vector
		 * @return support vector
		 */
		int32_t get_support_vector(int32_t idx);

		/** get alpha at given index
		 *
		 * @param idx index of alpha
		 * @return alpha
		 */
		float64_t get_alpha(int32_t idx);

		/** set support vector at given index to given value
		 *
		 * @param idx index of support vector
		 * @param val new value of support vector
		 * @return if operation was successful
		 */
		bool set_support_vector(int32_t idx, int32_t val);

		/** set alpha at given index to given value
		 *
		 * @param idx index of alpha vector
		 * @param val new value of alpha vector
		 * @return if operation was successful
		 */
		bool set_alpha(int32_t idx, float64_t val);

		/** get number of support vectors
		 *
		 * @return number of support vectors
		 */
		int32_t get_num_support_vectors();

		/** set alphas to given values
		 *
		 * @param alphas float vector with all alphas to set
		 */
		void set_alphas(SGVector<float64_t> alphas);

		/** set support vectors to given values
		 *
		 * @param svs integer vector with all support vectors indexes to set
		 */
		void set_support_vectors(SGVector<int32_t> svs);

		/** @return all support vectors */
		SGVector<int32_t> get_support_vectors();

		/** @return vector of alphas */
		SGVector<float64_t> get_alphas();

		/** create new model
		 *
		 * @param num number of alphas and support vectors in new model
		 */
		bool create_new_model(int32_t num);

		/** initialise kernel optimisation
		 *
		 * @return if operation was successful
		 */
		bool init_kernel_optimization();

		/** apply kernel machine to data
		 * for regression task
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

		/** apply kernel machine to data
		 * for binary classification task
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);

		/** apply kernel machine to one example
		 *
		 * @param num which example to apply to
		 * @return classified value
		 */
		virtual float64_t apply_one(int32_t num);

		/** apply example helper, used in threads
		 *
		 * @param p params of the thread
		 * @return nothing really
		 */
		static void* apply_helper(void* p);

		/** Trains a locked machine on a set of indices. Error if machine is
		 * not locked
		 *
		 * @param indices index vector (of locked features) that is used for training
		 * @return whether training was successful
		 */
		virtual bool train_locked(SGVector<index_t> indices);

		/** Applies a locked machine on a set of indices. Error if machine is
		 * not locked. Binary case
		 *
		 * @param indices index vector (of locked features) that is predicted
		 * @return resulting labels
		 */
		virtual CBinaryLabels* apply_locked_binary(SGVector<index_t> indices);

		/** Applies a locked machine on a set of indices. Error if machine is
		 * not locked. Binary case
		 *
		 * @param indices index vector (of locked features) that is predicted
		 * @return resulting labels
		 */
		virtual CRegressionLabels* apply_locked_regression(
				SGVector<index_t> indices);

		/** Applies a locked machine on a set of indices. Error if machine is
		 * not locked
		 *
		 * @param indices index vector (of locked features) that is predicted
		 * @return raw output of machine
		 */
		virtual SGVector<float64_t> apply_locked_get_output(
				SGVector<index_t> indices);

		/** Locks the machine on given labels and data. After this call, only
		 * train_locked and apply_locked may be called.
		 *
		 * Computes kernel matrix to speed up train/apply calls
		 *
		 * @param labs labels used for locking
		 * @param features features used for locking
		 */
		virtual void data_lock(CLabels* labs, CFeatures* features=NULL);

		/** Unlocks a locked machine and restores previous state */
		virtual void data_unlock();

		/** @return whether machine supports locking */
		virtual bool supports_locking() const;

	protected:

		/** apply get outputs
		 *
		 * @param data features to compute outputs
		 * @return outputs
		 */
		SGVector<float64_t> apply_get_outputs(CFeatures* data);

		/** Stores feature data of the SV indices and sets it to the lhs of the
		 * underlying kernel. Then, all SV indices are set to identity.
		 *
		 * May be overwritten by subclasses in case the model should be stored
		 * differently.
		 */
		virtual void store_model_features();

    private:
		/** register parameters and do misc init */
		void init();

	protected:
		/** kernel */
		CKernel* kernel;

		/** is filled with pre-computed custom kernel on data lock */
		CCustomKernel* m_custom_kernel;

		/** old kernel is stored here on data lock */
		CKernel* m_kernel_backup;

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
};
}
#endif /* _KERNEL_MACHINE_H__ */

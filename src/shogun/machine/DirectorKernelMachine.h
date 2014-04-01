/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */

#ifndef _DIRECTORKERNELMACHINE_H___
#define _DIRECTORKERNELMACHINE_H___

#include <shogun/lib/config.h>

#ifdef USE_SWIG_DIRECTORS
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/KernelMachine.h>

namespace shogun
{

#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CDirectorKernelMachine : public CKernelMachine
{
	public:
		/* default constructor */
		CDirectorKernelMachine()
		: CKernelMachine()
		{

		}

		/** Convenience constructor to initialize a trained kernel
		 * machine
		 *
		 * @param k kernel
		 * @param alphas vector of alpha weights
		 * @param svs indices of examples, i.e. i's for x_i
		 * @param b bias term
		 */
		CDirectorKernelMachine(CKernel* k, const SGVector<float64_t> alphas, const SGVector<int32_t> svs, float64_t b)
		: CKernelMachine(k, alphas, svs, b)
		{
		}

		/* destructor */
		virtual ~CDirectorKernelMachine()
		{

		}

		/** train machine
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data).
		 * If flag is set, model features will be stored after training.
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL)
		{
			return CKernelMachine::train(data);
		}

		virtual bool train_function(CFeatures* data=NULL)
		{
			SG_ERROR("Train function of Director Kernel Machine needs to be overridden.\n")
			return false;
		}

		/** apply machine to data
		 * if data is not specified apply to the current features
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data=NULL)
		{
			return CKernelMachine::apply(data);
		}

		/** apply machine to data in means of binary classification problem */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL)
		{
			return CKernelMachine::apply_binary(data);
		}

		/** apply machine to data in means of regression problem */
		virtual CRegressionLabels* apply_regression(CFeatures* data=NULL)
		{
			return CKernelMachine::apply_regression(data);
		}

		/** apply machine to data in means of multiclass classification problem */
		using CKernelMachine::apply_multiclass;

		/** apply kernel machine to one example
		 *
		 * @param num which example to apply to
		 * @return classified value
		 */
		virtual float64_t apply_one(int32_t num)
		{
			return CKernelMachine::apply_one(num);
		}

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(CLabels* lab)
		{
			CKernelMachine::set_labels(lab);
		}

		/** get labels
		 *
		 * @return labels
		 */
		virtual CLabels* get_labels()
		{
			return CKernelMachine::get_labels();
		}

		/** get classifier type
		 *
		 * @return classifier type NONE
		 */
		virtual EMachineType get_classifier_type() { return CT_DIRECTORKERNEL; }

		/** Setter for store-model-features-after-training flag
		 *
		 * @param store_model whether model should be stored after
		 * training
		 */
		virtual void set_store_model_features(bool store_model)
		{
			CKernelMachine::set_store_model_features(store_model);
		}

		/** Trains a locked machine on a set of indices. Error if machine is
		 * not locked
		 *
		 * NOT IMPLEMENTED
		 *
		 * @param indices index vector (of locked features) that is used for training
		 * @return whether training was successful
		 */
		virtual bool train_locked(SGVector<index_t> indices)
		{
			return CKernelMachine::train_locked(indices);
		}

		/** Applies a locked machine on a set of indices. Error if machine is
		 * not locked
		 *
		 * @param indices index vector (of locked features) that is predicted
		 */
		virtual CLabels* apply_locked(SGVector<index_t> indices)
		{
			return CKernelMachine::apply_locked(indices);
		}

		virtual CBinaryLabels* apply_locked_binary(SGVector<index_t> indices)
		{
			return CKernelMachine::apply_locked_binary(indices);
		}

		virtual CRegressionLabels* apply_locked_regression(
				SGVector<index_t> indices)
		{
			return CKernelMachine::apply_locked_regression(indices);
		}

		using CKernelMachine::apply_locked_multiclass;

		/** Applies a locked machine on a set of indices. Error if machine is
		 * not locked
		 *
		 * @param indices index vector (of locked features) that is predicted
		 * @return raw output of machine
		 */
		virtual SGVector<float64_t> apply_locked_get_output(
				SGVector<index_t> indices)
		{
			return CKernelMachine::apply_locked_get_output(indices);
		}

		/** Locks the machine on given labels and data. After this call, only
		 * train_locked and apply_locked may be called
		 *
		 * Only possible if supports_locking() returns true
		 *
		 * @param labs labels used for locking
		 * @param features features used for locking
		 */
		virtual void data_lock(CLabels* labs, CFeatures* features)
		{
			CKernelMachine::data_lock(labs, features);
		}

		/** Unlocks a locked machine and restores previous state */
		virtual void data_unlock()
		{
			CKernelMachine::data_unlock();
		}

		/** @return whether this machine supports locking */
		virtual bool supports_locking() const
		{
			return CKernelMachine::supports_locking();
		}

		//TODO change to pure virtual
		virtual EProblemType get_machine_problem_type() const
		{
			return CKernelMachine::get_machine_problem_type();
		}

		virtual const char* get_name() const { return "DirectorKernelMachine"; }

	protected:
		/** train machine
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL)
		{
			return train_function(data);
		}
};

}

#endif /* USE_SWIG_DIRECTORS */
#endif /* _DIRECTORKERNELMACHINE_H___ */

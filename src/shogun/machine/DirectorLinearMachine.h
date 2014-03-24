/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */

#ifndef _DIRECTORLINEARMACHINE_H___
#define _DIRECTORLINEARMACHINE_H___

#ifdef USE_SWIG_DIRECTORS
#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{

#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CDirectorLinearMachine : public CLinearMachine
{
	public:
		/* default constructor */
		CDirectorLinearMachine()
		: CLinearMachine()
		{

		}

		/* destructor */
		virtual ~CDirectorLinearMachine()
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
			return CLinearMachine::train(data);
		}

		virtual bool train_function(CFeatures* data=NULL)
		{
			SG_ERROR("Train function of Director Linear Machine needs to be overridden.\n")
			return false;
		}

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(CDotFeatures* feat)
		{
			CLinearMachine::set_features(feat);
		}

		/** get features
		 *
		 * @return features
		 */
		virtual CDotFeatures* get_features()
		{
			CLinearMachine::get_features();
		}

		/** apply machine to data
		 * if data is not specified apply to the current features
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data=NULL)
		{
			return CLinearMachine::apply(data);
		}

		/** apply machine to data in means of binary classification problem */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL)
		{
			return CLinearMachine::apply_binary(data);
		}

		/** apply machine to data in means of regression problem */
		virtual CRegressionLabels* apply_regression(CFeatures* data=NULL)
		{
			return CLinearMachine::apply_regression(data);
		}

		/** apply machine to data in means of multiclass classification problem */
		using CLinearMachine::apply_multiclass;

		virtual float64_t apply_one(int32_t vec_idx)
		{
			return CLinearMachine::apply_one(vec_idx);
		}

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(CLabels* lab)
		{
			CLinearMachine::set_labels(lab);
		}

		/** get labels
		 *
		 * @return labels
		 */
		virtual CLabels* get_labels()
		{
			return CLinearMachine::get_labels();
		}

		/** get classifier type
		 *
		 * @return classifier type NONE
		 */
		virtual EMachineType get_classifier_type() { return CT_DIRECTORLINEAR; }

		/** Setter for store-model-features-after-training flag
		 *
		 * @param store_model whether model should be stored after
		 * training
		 */
		virtual void set_store_model_features(bool store_model)
		{
			CLinearMachine::set_store_model_features(store_model);
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
			return CLinearMachine::train_locked(indices);
		}

		/** Applies a locked machine on a set of indices. Error if machine is
		 * not locked
		 *
		 * @param indices index vector (of locked features) that is predicted
		 */
		virtual CLabels* apply_locked(SGVector<index_t> indices)
		{
			return CLinearMachine::apply_locked(indices);
		}

		virtual CBinaryLabels* apply_locked_binary(SGVector<index_t> indices)
		{
			return CLinearMachine::apply_locked_binary(indices);
		}

		virtual CRegressionLabels* apply_locked_regression(
				SGVector<index_t> indices)
		{
			return CLinearMachine::apply_locked_regression(indices);
		}

		using CLinearMachine::apply_locked_multiclass;

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
			CLinearMachine::data_lock(labs, features);
		}

		/** Unlocks a locked machine and restores previous state */
		virtual void data_unlock()
		{
			CLinearMachine::data_unlock();
		}

		/** @return whether this machine supports locking */
		virtual bool supports_locking() const
		{
			return CLinearMachine::supports_locking();
		}

		//TODO change to pure virtual
		virtual EProblemType get_machine_problem_type() const
		{
			return CLinearMachine::get_machine_problem_type();
		}

		virtual const char* get_name() const { return "DirectorLinearMachine"; }

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
#endif /* _DIRECTORLINEARMACHINE_H___ */

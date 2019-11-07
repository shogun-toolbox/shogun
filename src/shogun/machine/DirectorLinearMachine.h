/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evgeniy Andreev, Tejas Jogi, Soeren Sonnenburg, Yuyu Zhang,
 *          Viktor Gal, Bjoern Esser
 */

#ifndef _DIRECTORLINEARMACHINE_H___
#define _DIRECTORLINEARMACHINE_H___

#include <shogun/lib/config.h>

#ifdef USE_SWIG_DIRECTORS
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{

#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class DirectorLinearMachine : public LinearMachine
{
	public:
		/* default constructor */
		DirectorLinearMachine()
		: LinearMachine()
		{

		}

		/* destructor */
		virtual ~DirectorLinearMachine()
		{

		}

		/** train machine
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data).
		 *
		 * @return whether training was successful
		 */
		virtual bool train(std::shared_ptr<Features> data=NULL)
		{
			return LinearMachine::train(data);
		}

		virtual bool train_function(std::shared_ptr<Features> data=NULL)
		{
			error("Train function of Director Linear Machine needs to be overridden.");
			return false;
		}

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(std::shared_ptr<DotFeatures> feat)
		{
			LinearMachine::set_features(feat);
		}

		/** get features
		 *
		 * @return features
		 */
		virtual std::shared_ptr<DotFeatures> get_features()
		{
			return LinearMachine::get_features();
		}

		/** apply machine to data
		 * if data is not specified apply to the current features
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual std::shared_ptr<Labels> apply(std::shared_ptr<Features> data=NULL)
		{
			return LinearMachine::apply(data);
		}

		/** apply machine to data in means of binary classification problem */
		virtual std::shared_ptr<BinaryLabels> apply_binary(std::shared_ptr<Features> data=NULL)
		{
			return LinearMachine::apply_binary(data);
		}

		/** apply machine to data in means of regression problem */
		virtual std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL)
		{
			return LinearMachine::apply_regression(data);
		}

		/** apply machine to data in means of multiclass classification problem */
		using LinearMachine::apply_multiclass;

		virtual float64_t apply_one(int32_t vec_idx)
		{
			return LinearMachine::apply_one(vec_idx);
		}

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(std::shared_ptr<Labels> lab)
		{
			LinearMachine::set_labels(lab);
		}

		/** get labels
		 *
		 * @return labels
		 */
		virtual std::shared_ptr<Labels> get_labels()
		{
			return LinearMachine::get_labels();
		}

		/** get classifier type
		 *
		 * @return classifier type NONE
		 */
		virtual EMachineType get_classifier_type() { return CT_DIRECTORLINEAR; }

		//TODO change to pure virtual
		virtual EProblemType get_machine_problem_type() const
		{
			return LinearMachine::get_machine_problem_type();
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
		virtual bool train_machine(std::shared_ptr<Features> data=NULL)
		{
			return train_function(data);
		}
};

}

#endif /* USE_SWIG_DIRECTORS */
#endif /* _DIRECTORLINEARMACHINE_H___ */

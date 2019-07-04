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
			return CLinearMachine::get_features();
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

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Tejas Jogi, Evgeniy Andreev, Soeren Sonnenburg, Yuyu Zhang, 
 *          Bjoern Esser
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

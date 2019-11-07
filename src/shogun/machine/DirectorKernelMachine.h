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
IGNORE_IN_CLASSLIST class DirectorKernelMachine : public KernelMachine
{
	public:
		/* default constructor */
		DirectorKernelMachine()
		: KernelMachine()
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
		DirectorKernelMachine(std::shared_ptr<Kernel> k, const SGVector<float64_t> alphas, const SGVector<int32_t> svs, float64_t b)
		: KernelMachine(k, alphas, svs, b)
		{
		}

		/* destructor */
		virtual ~DirectorKernelMachine()
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
			return KernelMachine::train(data);
		}

		virtual bool train_function(std::shared_ptr<Features> data=NULL)
		{
			error("Train function of Director Kernel Machine needs to be overridden.");
			return false;
		}

		/** apply machine to data
		 * if data is not specified apply to the current features
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual ::std::shared_ptr<Labels> apply(std::shared_ptr<Features> data=NULL)
		{
			return KernelMachine::apply(data);
		}

		/** apply machine to data in means of binary classification problem */
		virtual ::std::shared_ptr<BinaryLabels> apply_binary(std::shared_ptr<Features> data=NULL)
		{
			return KernelMachine::apply_binary(data);
		}

		/** apply machine to data in means of regression problem */
		virtual ::std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL)
		{
			return KernelMachine::apply_regression(data);
		}

		/** apply machine to data in means of multiclass classification problem */
		using KernelMachine::apply_multiclass;

		/** apply kernel machine to one example
		 *
		 * @param num which example to apply to
		 * @return classified value
		 */
		virtual float64_t apply_one(int32_t num)
		{
			return KernelMachine::apply_one(num);
		}

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(std::shared_ptr<Labels> lab)
		{
			KernelMachine::set_labels(lab);
		}

		/** get labels
		 *
		 * @return labels
		 */
		virtual std::shared_ptr<Labels> get_labels()
		{
			return KernelMachine::get_labels();
		}

		/** get classifier type
		 *
		 * @return classifier type NONE
		 */
		virtual EMachineType get_classifier_type() { return CT_DIRECTORKERNEL; }

		//TODO change to pure virtual
		virtual EProblemType get_machine_problem_type() const
		{
			return KernelMachine::get_machine_problem_type();
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
		virtual bool train_machine(std::shared_ptr<Features> data=NULL)
		{
			return train_function(data);
		}
};

}

#endif /* USE_SWIG_DIRECTORS */
#endif /* _DIRECTORKERNELMACHINE_H___ */

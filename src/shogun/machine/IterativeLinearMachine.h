/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */

#ifndef _ITERATIVELINEARMACHINE_H__
#define _ITERATIVELINEARMACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	class CFeatures;
	class CLabels;

	class CIterativeLinearMachine : public CLinearMachine
	{
	public:
		/** Default constructor */
		CIterativeLinearMachine();
		virtual ~CIterativeLinearMachine();

		/** Train classifier
		  *
		  * @param data training data (parameter can be avoided if distance or
		  * kernel-based classifiers are used and distance/kernels are
		  * initialized with train data)
		  *
		  * @return whether training was successful
		  */
		virtual bool train_machine(CFeatures* data = NULL);

		/** Continue Training
		  *
		  * Only possible if a previous CMachine::train call was prematurely
		  * stopped.
		  * Throws an error otherwise
		  *
		  * @return whether continue training was successful
		  */
		virtual bool continue_train();

	protected:
		/** Can be overloaded by sublcasses to implement custom single
		  *iterations of training loop.
		  */
		virtual void iteration() = 0;

		/** Can be overloaded in subclasses to initialize the model for training
		  */
		virtual void init_model(CFeatures* data = NULL) = 0;

		/** Maximum Iterations */
		int64_t m_max_iterations;
		/** Current iteration of training loop */
		int64_t m_current_iteration;
		/** Completion status */
		bool m_complete;
		/** Features */
		CFeatures* m_continue_features;
	};
}
#endif

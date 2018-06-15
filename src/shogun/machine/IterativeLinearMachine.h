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

	/** @brief Class IterativeLinearMachine implements an iterative model
	 * whose training can be prematurely stopped, and in particular
	 * resumed, anytime.
	 */
	class CIterativeLinearMachine : public CLinearMachine
	{
	public:
		/** Default constructor */
		CIterativeLinearMachine();
		virtual ~CIterativeLinearMachine();

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
		virtual bool train_machine(CFeatures* data = NULL);

		/** To be overloaded by sublcasses to implement custom single
		  * iterations of training loop.
		  */
		virtual void iteration() = 0;

		/** To be overloaded in subclasses to initialize the model for training
		  */
		virtual void init_model(CFeatures* data = NULL) = 0;

		/** Can be overloaded in subclasses to show more information
		  * and/or clean up states
		  */
		virtual void end_training();

		/** Maximum Iterations */
		int32_t m_max_iterations;
		/** Current iteration of training loop */
		int32_t m_current_iteration;
		/** Completion status */
		bool m_complete;
		/** Features */
		CFeatures* m_continue_features;
	};
}
#endif

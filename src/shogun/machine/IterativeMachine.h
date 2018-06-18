/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */

#ifndef _ITERATIVEMACHINE_H__
#define _ITERATIVEMACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	class CFeatures;
	class CLabels;

#define SG_ADD8(param, name, description, ms_available)                        \
	{                                                                          \
		this->m_parameters->add(param, name, description);                     \
		this->watch_param(                                                     \
		    name, param,                                                       \
		    AnyParameterProperties(                                            \
		        description, ms_available, GRADIENT_NOT_AVAILABLE));           \
		if (ms_available)                                                      \
			this->m_model_selection_parameters->add(param, name, description); \
	}

	/** @brief Class IterativeLinearMachine implements an iterative model
	 * whose training can be prematurely stopped, and in particular
	 * resumed, anytime.
	 */
	template <class T>
	class CIterativeMachine : public T
	{
	public:
		/** Default constructor */
		CIterativeMachine() : T()
		{
			m_current_iteration = 0;
			m_complete = false;

			SG_ADD8(
			    &m_current_iteration, "current_iteration",
			    "Current Iteration of training", MS_NOT_AVAILABLE);
			SG_ADD8(
			    &m_max_iterations, "max_iterations",
			    "Maximum number of Iterations", MS_AVAILABLE);
			SG_ADD8(
			    &m_complete, "complete", "Convergence status",
			    MS_NOT_AVAILABLE);
		}

		virtual ~CIterativeMachine()
		{
		}

		/** Continue Training
		  *
		  * Only possible if a previous CMachine::train call was prematurely
		  * stopped.
		  * Throws an error otherwise
		  *
		  * @return whether continue training was successful
		  */
		virtual bool continue_train()
		{
			this->reset_computation_variables();
			while (m_current_iteration < m_max_iterations && !m_complete)
			{
				if (this->cancel_computation())
					break;
				this->pause_computation();
				iteration();
				m_current_iteration++;
			}

			if (m_complete)
				SG_SINFO(
				    "%s converged after %d iterations.\n", this->get_name(),
				    m_current_iteration)
			else if (!m_complete && m_current_iteration == m_max_iterations)
			{
				SG_SWARNING(
				    "%s did not converge after the maximum number of %d "
				    "iterations.\n",
				    this->get_name(), m_current_iteration)

				this->end_training();
			}
			return m_complete;
		}

	protected:
		virtual bool train_machine(CFeatures* data = NULL)
		{
			m_current_iteration = 0;
			m_complete = false;
			init_model(data);
			return continue_train();
		}

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
		virtual void end_training()
		{
		}

		/** Maximum Iterations */
		int32_t m_max_iterations;
		/** Current iteration of training loop */
		int32_t m_current_iteration;
		/** Completion status */
		bool m_complete;
	};
}
#endif

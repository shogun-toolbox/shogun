/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */

#ifndef _ITERATIVEMACHINE_H__
#define _ITERATIVEMACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/base/progress.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	class CFeatures;
	class CLabels;

	/** @brief Mix-in class that implements an iterative model
	 * whose training can be prematurely stopped, and in particular be
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
			m_continue_features = nullptr;

			SG_ADD(
			    &m_current_iteration, "current_iteration",
			    "Current Iteration of training");
			SG_ADD(
			    &m_max_iterations, "max_iterations",
			    "Maximum number of Iterations", ParameterProperties::HYPER);
			SG_ADD(
			    &m_complete, "complete", "Convergence status");
			SG_ADD(
			    &m_continue_features, "continue_features", "Continue Features");
		}
		virtual ~CIterativeMachine()
		{
			SG_UNREF(m_continue_features);
		}

		/** Returns convergence status */
		bool is_complete()
		{
			return m_complete;
		}

		virtual bool continue_train()
		{
			this->reset_computation_variables();
			this->put("features", m_continue_features);

			auto pb = SG_PROGRESS(range(m_max_iterations));
			while (m_current_iteration < m_max_iterations && !m_complete)
			{
				COMPUTATION_CONTROLLERS
				iteration();
				m_current_iteration++;
				pb.print_progress();
			}
			pb.complete();

			if (m_complete)
			{
				SG_SINFO(
				    "%s converged after %d iterations.\n", this->get_name(),
				    m_current_iteration)

				this->end_training();
			}
			else if (!m_complete && m_current_iteration >= m_max_iterations)
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
			train_machine(data, T::m_labels);
			return is_complete();
		}

		virtual void
		train_machine(CFeatures* features, CLabels* labels)
		{
			if (features)
			{
				SG_REF(features);
				SG_UNREF(m_continue_features);
				m_continue_features = features;
			}
			T::set_labels(labels);
			m_current_iteration = 0;
			m_complete = false;
			init_model(features, labels);
			continue_train();
		}

		/** To be overloaded by sublcasses to implement custom single
		  * iterations of training loop.
		  */
		virtual void iteration() = 0;

		/** To be overloaded in subclasses to initialize the model for training
		  */
		virtual void init_model(CFeatures* data = NULL)
		{
			SG_SNOTIMPLEMENTED
		}

		// FIXME: this is a temporary implementation during refactor
		virtual void init_model(CFeatures* features, CLabels* labels)
		{
			SG_SNOTIMPLEMENTED
		}

		/** Can be overloaded in subclasses to show more information
		  * and/or clean up states
		  */
		virtual void end_training()
		{
		}

		/** Stores features to continue training */
		CFeatures* m_continue_features;
		/** Maximum Iterations */
		int32_t m_max_iterations;
		/** Current iteration of training loop */
		int32_t m_current_iteration;
		/** Completion status */
		bool m_complete;
	};
}
#endif
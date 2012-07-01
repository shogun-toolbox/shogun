/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (W) 2012 Sergey Lisitsyn
 */

#ifndef PEGASOS_SVM_H_
#define PEGASOS_SVM_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/optimization/pegasos/pegasos_optimize.h>

namespace shogun
{

/** @brief */
class CPegasosSVM : public CLinearMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor  */
		CPegasosSVM();

		/** constructor
		 *
		 * @param lambda constant lambda
		 * @param traindat training features
		 * @param trainlab training labels
		 */
		CPegasosSVM(
			float64_t lambda, CDotFeatures* traindat,
			CLabels* trainlab);

		/** destructor */
		virtual ~CPegasosSVM();

		/** set lambda
		 *
		 * @param lambda lambda
		 */
		inline void set_lambda(float64_t lambda) { m_lambda = lambda; }

		/** get lambda
		 *
		 * @return lambda
		 */
		inline float64_t get_lambda() { return m_lambda; }

		/** @return object name */
		inline virtual const char* get_name() const { return "PegasosSVM"; }

		/** get the maximum number of iterations solver is allowed to do */
		inline int32_t get_max_iterations()
		{
			return m_max_iterations;
		}

		/** set the maximum number of iterations solver is allowed to do */
		inline void set_max_iterations(int32_t max_iter=1000)
		{
			m_max_iterations=max_iter;
		}

	private:

		/** init */
		void init();

	protected:
		/** train linear SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	protected:
		/** lambda */
		float64_t m_lambda;
		/** maximum number of iterations */
		int32_t m_max_iterations;
};

} /* namespace shogun  */
#endif 

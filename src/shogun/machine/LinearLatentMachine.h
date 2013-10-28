/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTLINEARMACHINE_H__
#define __LATENTLINEARMACHINE_H__

#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/latent/LatentModel.h>

namespace shogun
{
	/** @brief abstract implementaion of Linear Machine with latent variable
	 * This is the base implementation of all linear machines with latent variable.
	 */
	class CLinearLatentMachine: public CLinearMachine
	{

		public:

			/** problem type */
			MACHINE_PROBLEM_TYPE(PT_LATENT);

			/** default contstructor */
			CLinearLatentMachine();

			/** constructor
			 *
			 * @param model the user defined CLatentModel
			 * @param C regularisation constant
			 */
			CLinearLatentMachine(CLatentModel* model, float64_t C);

			virtual ~CLinearLatentMachine();

			/** apply linear machine to data set before
			 *
			 * @return classified labels
			 */
			virtual CLatentLabels* apply_latent() = 0;

			/** apply linear machine to data
			 *
			 * @param data (test)data to be classified
			 * @return classified labels
			 */
			virtual CLatentLabels* apply_latent(CFeatures* data);

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LinearLatentMachine"; }

			/** set epsilon
			 *
			 * @param eps new epsilon
			 */
			inline void set_epsilon(float64_t eps) { m_epsilon=eps; }

			/** get epsilon
			 *
			 * @return epsilon
			 */
			inline float64_t get_epsilon() { return m_epsilon; }

			/** set C
			 *
			 * @param c new C constant
			 */
			inline void set_C(float64_t c)
			{
				m_C=c;
			}

			/** get C
			 *
			 * @return C
			 */
			inline float64_t get_C() { return m_C; }

			/** set maximum iterations
			 *
			 * @param iter new maximum iteration value
			 */
			inline void set_max_iterations(int32_t iter) { m_max_iter = iter; }

			/** get maximum iterations value
			 *
			 * @return maximum iterations
			 */
			inline int32_t get_max_iterations() { return m_max_iter; }

			/** set latent model
			 *
			 * @param latent_model user defined latent model
			 */
			void set_model(CLatentModel* latent_model);

		protected:
			virtual bool train_machine(CFeatures* data=NULL);

			/** inner loop of the latent machine
			 *
			 * @param cooling_eps epsilon
			 *
			 * @return primal objective value
			 */
			virtual float64_t do_inner_loop(float64_t cooling_eps)=0;

			virtual bool train_require_labels() const { return false; }

		protected:
			/** user supplied latent model */
			CLatentModel* m_model;
			/** C */
			float64_t m_C;
			/** epsilon */
			float64_t m_epsilon;
			/** max iterations */
			int32_t m_max_iter;
			/** current iteration */
			int32_t m_cur_iter;

		private:
			/** initalize the values to default values */
			void init();
	};
}

#endif /* __LATENTLINEARMACHINE_H__ */


/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Yuyu Zhang, Thoralf Klein, Sergey Lisitsyn,
 *          Bjoern Esser, Soeren Sonnenburg
 */

#ifndef __LATENTLINEARMACHINE_H__
#define __LATENTLINEARMACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	class Features;
	class LatentLabels;
	class LatentModel;

	/** @brief abstract implementaion of Linear Machine with latent variable
	 * This is the base implementation of all linear machines with latent variable.
	 */
	class LinearLatentMachine: public LinearMachine
	{

		public:

			/** problem type */
			MACHINE_PROBLEM_TYPE(PT_LATENT);

			/** default contstructor */
			LinearLatentMachine();

			/** constructor
			 *
			 * @param model the user defined LatentModel
			 * @param C regularisation constant
			 */
			LinearLatentMachine(std::shared_ptr<LatentModel> model, float64_t C);

			virtual ~LinearLatentMachine();

			/** apply linear machine to data set before
			 *
			 * @return classified labels
			 */
			virtual std::shared_ptr<LatentLabels> apply_latent() = 0;

			/** apply linear machine to data
			 *
			 * @param data (test)data to be classified
			 * @return classified labels
			 */
			virtual std::shared_ptr<LatentLabels> apply_latent(std::shared_ptr<Features> data);

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
			void set_model(const std::shared_ptr<LatentModel>& latent_model);

			virtual bool train_require_labels() const
			{
				return false;
			}

		protected:
			virtual bool train_machine(std::shared_ptr<Features> data=NULL);

			/** inner loop of the latent machine
			 *
			 * @param cooling_eps epsilon
			 *
			 * @return primal objective value
			 */
			virtual float64_t do_inner_loop(float64_t cooling_eps)=0;

		protected:
			/** user supplied latent model */
			std::shared_ptr<LatentModel> m_model;
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

#ifndef __LATENTSVM_H__
#define __LATENTSVM_H__

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/machine/LinearLatentMachine.h>

namespace shogun
{
	class LatentModel;

	/** @brief LatentSVM class
	 * Latent SVM implementation based on [1].
	 * For optimization this implementation uses SVMOcas.
	 *
	 * User must provide a her own CLatentModel which implements the PSI(x_i,h_i)
	 * function for the given problem.
	 *
	 * [1] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan,
	 *  "Object detection with discriminatively trained part-based models,"
	 *  Pattern Analysis and Machine Intelligence,
	 *  IEEE Transactions on, vol. 32, no. 9, pp. 1627-1645, 2010.
	 *
	 */
	class CLatentSVM: public CLinearLatentMachine
	{
		public:
			/** default contstructor */
			CLatentSVM();

			/** constructor
			 *
			 * @param model the user defined CLatentModel object.
			 * @param C regularization constant
			 */
			CLatentSVM(CLatentModel* model, float64_t C);

			virtual ~CLatentSVM();

			/** apply linear machine to all examples
			 *
			 * @return resulting labels
			 */
			virtual CLatentLabels* apply_latent();

			using CLinearLatentMachine::apply_latent;

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentSVM"; }

		protected:
			/** inner loop of the latent machine
			 *
			 * The optimization part after finding the argmax_h for the
			 * positive examples in the outter loop. It uses SVMOcas for
			 * finding the cutting plane.
			 *
			 * @param cooling_eps epsilon
			 * @return primal objective value
			 */
			virtual float64_t do_inner_loop(float64_t cooling_eps);
	};
}

#endif /* __LATENTSVM_H__ */

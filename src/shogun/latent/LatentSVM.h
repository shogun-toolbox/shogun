/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTSVM_H__
#define __LATENTSVM_H__

#include <shogun/lib/common.h>
#include <shogun/machine/LinearLatentMachine.h>

namespace shogun
{
	class CLatentSVM: public CLinearLatentMachine
	{

		public:

			/** default contstructor */
			CLatentSVM();

			/** constructor
			 *
			 * @param C constant C
			 * @param traindat training features
			 * @param trainlab labels for training features
			 */
			CLatentSVM(CLatentModel* model, float64_t C);

			virtual ~CLatentSVM();

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentSVM"; }

		protected:
			/** inner loop of the latent machine
			 *
			 * @param cooling_eps epsilon
			 *
			 * @return primal objective value
			 */
			virtual float64_t do_inner_loop(float64_t cooling_eps);
	};
}

#endif /* __LATENTSVM_H__ */


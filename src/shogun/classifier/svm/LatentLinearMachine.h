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
#include <shogun/features/LatentLabels.h>

namespace shogun
{
	/* function pointer to the latent variable minimalization */
	typedef CLatentFeatures* (*minimizeLatent) (CLatentFeatures *lFeatures, void* userData);
	
	class CLatentLinearMachine : public CLinearMachine
	{
		
		public:
			CLatentLinearMachine (minLatent usrFunc);
		
			virtual ~CLatentLinearMachine ();
		
			/** apply linear machine to all examples
			 *
			 * @return resulting labels
			 */
			virtual CLatentLabels* apply ();
		
			/** apply linear machine to data
			 *
			 * @param data (test)data to be classified
			 * @return classified labels
			 */
			virtual CLatentLabels* apply (CFeatures* data);
		
			/** get features
			 *
			 * @return features
			 */
			virtual CDotFeatures* get_features() { SG_REF(features); return features; }

			/** Returns the name of the SGSerializable instance.  It MUST BE
			 *  the CLASS NAME without the prefixed `C'.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentLinearMachine"; }
			
		private:
			minimizeLatent handleLatent;
	};
}

#endif /* __LATENTLINEARMACHINE_H__ */

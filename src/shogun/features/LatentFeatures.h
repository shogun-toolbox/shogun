/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTFEATURES_H__
#define __LATENTFEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/lib/Data.h>

namespace shogun
{
	/** @brief Latent Features class
	 * The class if for representing features for latent learning, e.g. LatentSVM.
	 * It's basically a very generic way of storing features of any (user-defined) form
	 * based on CData.
	 */
	class CLatentFeatures : public CFeatures
	{
		public:
			/** default constructor */
			CLatentFeatures();

			/** constructor
			 *
			 * @param num_samples the number of examples the object will contain
			 */
			CLatentFeatures(int32_t num_samples);

			virtual ~CLatentFeatures();

			/** Copy-constructor
			 *
			 * @return the copy of the given object
			 */
			virtual CFeatures* duplicate() const;

			/** get feature type
			 *
			 * @return templated feature type
			 */
			virtual EFeatureType get_feature_type() const;

			/** get feature class
			 *
			 * @return feature class
			 */
			virtual EFeatureClass get_feature_class() const;

			/** get number of examples
			 *
			 * @return number of examples/vectors (possibly of subset, if implemented)
			 */
			virtual int32_t get_num_vectors() const;

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentFeatures"; }

			/** add latent example
			 *
			 * @param example the user defined CData
			 */
			bool add_sample(CData* example);

			/** get latent example
			 *
			 * @param idx index of the required example
			 * @return the user defined CData at the given index
			 */
			CData* get_sample(index_t idx);

			/** helper method used to specialize a base class instance
			 *
			 * @param base_feats its dynamic type must be CLatentFeatures
			 */
			static CLatentFeatures* obtain_from_generic(CFeatures* base_feats);
		protected:
			/** array of CData */
			CDynamicObjectArray* m_samples;

		private:
			/** init function for the object */
			void init();
	};
}

#endif /* __LATENTFEATURES_H__ */


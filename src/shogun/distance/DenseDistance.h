/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DENSEDISTANCE_H___
#define _DENSEDISTANCE_H___

#include <shogun/lib/config.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
/** @brief template class DenseDistance */
template <class ST> class CDenseDistance : public CDistance
{
	public:
		/** default constructor */
		CDenseDistance() : CDistance() {}

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** get feature class the distance can deal with
		 *
		 * @return feature class DENSE
		 */
		virtual EFeatureClass get_feature_class() { return C_DENSE; }

		/** get feature type the distance can deal with
		 *
		 * @return template-specific feature type
		 */
		virtual EFeatureType get_feature_type();

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		virtual const char* get_name() const {
			return "DenseDistance"; }

		/** get distance type we are
		 *
		 * abstrace base method
		 *
		 * @return distance type
		 */
		virtual EDistanceType get_distance_type()=0;
};
} // namespace shogun
#endif

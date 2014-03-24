/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _REALDISTANCE_H__
#define _REALDISTANCE_H__

#include <shogun/lib/config.h>
#include <shogun/distance/DenseDistance.h>
#include <shogun/lib/common.h>

namespace shogun
{
/** @brief class RealDistance */
class CRealDistance : public CDenseDistance<float64_t>
{
public:
	/** default constructor */
	CRealDistance() : CDenseDistance<float64_t>() {}

	/** init distance
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if init was successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r)
	{
		CDenseDistance<float64_t>::init(l,r);

		ASSERT(l->get_feature_type()==F_DREAL)
		ASSERT(r->get_feature_type()==F_DREAL)

		return true;
	}

	/** get feature type the distance can deal with
	 *
	 * @return feature type DREAL
	 */
	virtual EFeatureType get_feature_type() { return F_DREAL; }

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	virtual const char* get_name() const { return "RealDistance"; }

	/** cleanup distance
	 *
	 * abstract base method
	 */
	virtual void cleanup()=0;

	/** get distance type we are
	 *
	 * abstrace base method
	 *
	 * @return distance type
	 */
	virtual EDistanceType get_distance_type()=0 ;

protected:
	/// compute distance function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual float64_t compute(int32_t x, int32_t y)=0;
};
} // namespace shogun
#endif

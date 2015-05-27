/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Gehl
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MANHATTANWORDDISTANCE_H___
#define _MANHATTANWORDDISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/distance/StringDistance.h>

using namespace shogun;

namespace distance
{
/** @brief class ManhattanWordDistance */
class CManhattanWordDistance: public CStringDistance<uint16_t>
{
	public:
		/** default constructor */
		CManhattanWordDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CManhattanWordDistance(CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r);
		virtual ~CManhattanWordDistance();

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** cleanup distance */
		virtual void cleanup();

		/** get distance type we are
		 *
		 * @return distance type MANHATTANWORD
		 */
		virtual EDistanceType get_distance_type() { return D_MANHATTANWORD; }

		/** get name of the distance
		 *
		 * @return name ManhattanWord
		 */
		virtual const char* get_name() const { return "ManhattanWordDistance"; }

	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b);
};
} // namespace shogun
#endif /* _MANHATTANWORDDISTANCE_H___ */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Christian Gehl
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST
 */

#ifndef _BRAYCURTISDISTANCE_H___
#define _BRAYCURTISDISTANCE_H___

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class Bray-Curtis distance
 *
 * The Bray-Curtis distance (Sorensen distance) is similar to the
 * Manhattan distance with normalization.
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x}') = \frac{\sum_{i=1}^{n}|x_{i}-x'_{i}|}{\sum_{i=1}^{n}|x_{i}
 *  +x'_{i}|} \quad x,x' \in R^{n}
 *  \f]
 *
 */
class CBrayCurtisDistance: public CDenseDistance<float64_t>
{
	public:
		/** default constructor */
		CBrayCurtisDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CBrayCurtisDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r);
		virtual ~CBrayCurtisDistance();

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
		 * @return distance type BRAYCURTIS
		 */
		virtual EDistanceType get_distance_type() { return D_BRAYCURTIS; }

		/** get name of the distance
		 *
		 * @return name Bray-Curtis distance
		 */
		virtual const char* get_name() const { return "BrayCurtisDistance"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};
} // namespace shogun
#endif /* _BRAYCURTISDISTANCE_H___ */

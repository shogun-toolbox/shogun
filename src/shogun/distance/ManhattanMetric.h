/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MANHATTANMETRIC_H___
#define _MANHATTANMETRIC_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

using namespace shogun;

namespace distance
{
/** @brief class ManhattanMetric
 *
 * The Manhattan distance (city block distance,\f$L_{1}\f$ norm, rectilinear
 * distance or taxi cab metric ) is a special case
 * of general Minkowski metric and computes the absolute differences
 * between the feature dimensions of two data points.
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = \sum_{i=1}^{n} |\bf{x_{i}}-\bf{x'_{i}}| \quad
 *  \bf{x},\bf{x'} \in R^{n}
 * \f]
 *
 * @see CMinkowskiMetric
 * @see <a href="http://en.wikipedia.org/wiki/Manhattan_distance">
 * Wikipedia: Manhattan distance</a>
 */
class CManhattanMetric: public CDenseDistance<float64_t>
{
	public:
		/** default constructor */
		CManhattanMetric();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CManhattanMetric(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r);
		virtual ~CManhattanMetric();

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
		 * @return distance type MANHATTAN
		 */
		virtual EDistanceType get_distance_type() { return D_MANHATTAN; }

		/** get name of the distance
		 *
		 * @return name Manhattan-Metric
		 */
		virtual const char* get_name() const { return "ManhattanMetric"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};

} // namespace shogun
#endif /* _MANHATTANMETRIC_H___ */


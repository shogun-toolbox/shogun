/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Christian Gehl
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST
 */

#ifndef _TANIMOTODISTANCE_H___
#define _TANIMOTODISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class Tanimoto coefficient
 *
 * The Tanimoto distance/coefficient (extended Jaccard coefficient)
 * is obtained by extending the cosine similarity.
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = \frac{\sum_{i=1}^{n}x_{i}x'_{i}}{
 *  \sum_{i=1}^{n}x_{i}x_{i}x'_{i}x'_{i}-x_{i}x'_{i}}
 *  /quad x,x' /in R^{n}
 * \f]
 *
 * @see <a href="http://en.wikipedia.org/wiki/Jaccard_index">Wikipedia:
 * Tanimoto coefficient</a>
 * @see CCosineDistance
 */
class CTanimotoDistance: public CDenseDistance<float64_t>
{
	public:
		/** default constructor */
		CTanimotoDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CTanimotoDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r);
		virtual ~CTanimotoDistance();

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
		 * @return distance type TANIMOTO
		 */
		virtual EDistanceType get_distance_type() { return D_TANIMOTO; }

		/** get name of the distance
		 *
		 * @return name Tanimoto coefficient/distance
		 */
		virtual const char* get_name() const { return "TanimotoDistance"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};
} // namespace shogun
#endif /* _TANIMOTODISTANCE_H___ */

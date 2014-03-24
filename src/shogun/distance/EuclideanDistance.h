/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _EUCLIDEANDISTANCE_H__
#define _EUCLIDEANDISTANCE_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/distance/RealDistance.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief class EuclideanDistance
 *
 * The familiar Euclidean distance for real valued features computes
 * the square root of the sum of squared disparity between the
 * corresponding feature dimensions of two data points.
 *
 * \f[\displaystyle
 *  d({\bf x},{\bf x'})= \sqrt{\sum_{i=0}^{n}|{\bf x_i}-{\bf x'_i}|^2}
 * \f]
 *
 * This special case of Minkowski metric is invariant to an arbitrary
 * translation or rotation in feature space.
 *
 * The Euclidean Squared distance does not take the square root:
 *
 * \f[\displaystyle
 *  d({\bf x},{\bf x'})= \sum_{i=0}^{n}|{\bf x_i}-{\bf x'_i}|^2
 * \f]
 *
 * @see CMinkowskiMetric
 * @see <a href="http://en.wikipedia.org/wiki/Distance#Distance_in_Euclidean_space">
 * Wikipedia: Distance in Euclidean space</a>
 */
class CEuclideanDistance: public CRealDistance
{
	public:
		/** default constructor */
		CEuclideanDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CEuclideanDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r);
		virtual ~CEuclideanDistance();

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
		 * @return distance type EUCLIDEAN
		 */
		virtual EDistanceType get_distance_type() { return D_EUCLIDEAN; }

		/** get feature type the distance can deal with
		 *
		 * @return feature type DREAL
		 */
		virtual EFeatureType get_feature_type() { return F_DREAL; }

		/** get name of the distance
		 *
		 * @return name Euclidean
		 */
		virtual const char* get_name() const { return "EuclideanDistance"; }

		/** disable application of sqrt on matrix computation
		 * the matrix can then also be named norm squared
		 *
		 * @return if application of sqrt is disabled
		 */
		virtual bool get_disable_sqrt() { return disable_sqrt; };

		/** disable application of sqrt on matrix computation
		 * the matrix can then also be named norm squared
		 *
		 * @param state new disable_sqrt
		 */
		virtual void set_disable_sqrt(bool state) { disable_sqrt=state; };

		/** compute the distance between lhs feature vector a
		 *  and rhs feature vector b. The computation of the
		 *  distance stops if the intermediate result is
		 *  larger than upper_bound. This is useful to use
		 *  with John Langford's Cover Tree
		 *
		 *  @param idx_a feature vector a at idx_a
		 *  @param idx_b feature vector b at idx_b
		 *  @param upper_bound value above which the computation
		 *  halts
		 *  @return distance value or upper_bound
		 */
		virtual float64_t distance_upper_bounded(int32_t idx_a, int32_t idx_b, float64_t upper_bound);

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		void init();

	protected:
		/** if application of sqrt on matrix computation is disabled */
		bool disable_sqrt;
};

} // namespace shogun
#endif /* _EUCLIDEANDISTANCE_H__ */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _MAHALANOBISDISTANCE_H__
#define _MAHALANOBISDISTANCE_H__

#ifdef HAVE_LAPACK

#include <shogun/lib/common.h>
#include <shogun/distance/RealDistance.h>
#include <shogun/features/SimpleFeatures.h>

namespace shogun
{
/** @brief class MahalanobisDistance 
 *
 * The Mahalanobis distance for real valued features computes the distance
 * between a feature vector and a distribution of features characterized by its 
 * mean and covariance.
 *
 * \f[\displaystyle
 *  D = \sqrt{ (x_i - \mu)' \Sigma^{-1} (x_i - \mu)  }
 * \f]
 * 
 * The Mahalanobis Squared distance does not take the square root:
 *
 * \f[\displaystyle
 *  D = (x_i - \mu)' \Sigma^{-1} (x_i - \mu)
 * \f]
 *
 * @see <a href="en.wikipedia.org/wiki/Mahalanobis_distance"> 
 * Wikipedia: Mahalanobis Distance</a>                   
 */
class CMahalanobisDistance: public CRealDistance
{
	public:
		/** default constructor */
		CMahalanobisDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CMahalanobisDistance(CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r);
		virtual ~CMahalanobisDistance();

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
		 * @return distance type MAHALANOBIS
		 */
		virtual EDistanceType get_distance_type() { return D_MAHALANOBIS; }

		/** get feature type the distance can deal with
		 *
		 * @return feature type DREAL
		 */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }

		/** get name of the distance
		 *
		 * @return name Mahalanobis
		 */
		virtual const char* get_name() const { return "MahalanobisDistance"; }

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

		/** whether the distance is between the mean of lhs and a vector of 
		 * rhs
		 *
		 * @return if the mean of lhs is used to obtain the distance
		 */
		virtual bool get_use_mean() { return use_mean; };

		/** whether the distance is between the mean of lhs and a vector of 
		 * rhs
		 *
		 * @param state new use_mean
		 */
		virtual void set_use_mean(bool state) { use_mean=state; };

	protected:
		/// compute Mahalanobis distance between a feature vector of lhs
		/// to a feature vector of rhs
		/// if use_mean then idx_a is not used and the distance 
		/// computed is between a feature vector of rhs and the 
		/// distribution lhs
		///
		/// @param idx_a index of the feature vector in lhs
		/// @param idx_b index of the feature vector in rhs
		/// @return value of the Mahalanobis distance
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		void init();

	protected:
		/** if application of sqrt on matrix computation is disabled */
		bool disable_sqrt;

		/** whether the features lhs and rhs have exactly the same values */
		bool use_mean;

		/** vector mean of the lhs feature vectors */
		SGVector<float64_t> mean;
		/** inverse of the covariance matrix of lhs feature vectors */
		SGMatrix<float64_t> icov;
};

} // namespace shogun
#endif /* HAVE_LAPACK */
#endif /* _MAHALANOBISDISTANCE_H__ */

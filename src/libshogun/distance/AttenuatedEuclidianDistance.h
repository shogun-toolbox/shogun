/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Miguel Angel Bautista
 * Copyright (C) 2011 Berlin Institute of Technology and Max Planck Society
 */

#ifndef _ATTENUATEDEUCLIDIANDISTANCE_H__
#define _ATTENUATEDEUCLIDIANDISTANCE_H__

#include "lib/common.h"
#include "distance/RealDistance.h"
#include "features/SimpleFeatures.h"

namespace shogun
{
/** @brief class AttenuatedEuclidianDistance
 *
 * The adaptation of the familiar Euclidean Distance, to be used in
 * ternary ECOC designs. This adaptation computes the Euclidean distance
 * between two vectors ignoring those positions of either of the vectors
 * valued as 0. Note that this might make sense only in the Decoding
 *  step of the ECOC framework, since the 0 value denotes that a certain category is
 *  ignored.
 * 
 * This distance was proposed by
 * S. Escalera, O. Pujol, P.Radeva in On the decoding process in ternary error-correcting output codes, 
 * Transactions in Pattern Analysis and Machine Intelligence 99 (1).
 *
 * \f[\displaystyle
 *  d({\bf x},{\bf x'})= \sqrt{\sum_{i=0}^{n}|x_i||x'_i|{\bf x_i}-{\bf x'_i}|^2}
 * \f]
 *                  
 */
class CAttenuatedEuclidianDistance: public CRealDistance
{
	public:
		/** default constructor */
		CAttenuatedEuclidianDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CAttenuatedEuclidianDistance(CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r);
		virtual ~CAttenuatedEuclidianDistance();

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
		 * @return distance type EUCLIDIAN
		 */
		virtual EDistanceType get_distance_type() { return D_ATTENUATEDEUCLIDIAN; }

		/** get feature type the distance can deal with
		 *
		 * @return feature type DREAL
		 */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }

		/** get name of the distance
		 *
		 * @return name Euclidian
		 */
		virtual const char* get_name() const { return "AttenuatedEuclidianDistance"; }

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
#endif /* _ATTENUATEDEUCLIDIANDISTANCE_H__ */

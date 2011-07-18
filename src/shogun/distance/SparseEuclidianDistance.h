/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEEUCLIDIANDISTANCE_H__
#define _SPARSEEUCLIDIANDISTANCE_H__

#include <shogun/lib/common.h>
#include <shogun/distance/SparseDistance.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
	template <class T> class CSparseFeatures;
/** @brief class SparseEucldianDistance */
class CSparseEuclidianDistance: public CSparseDistance<float64_t>
{
	public:
		/** default constructor */
		CSparseEuclidianDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CSparseEuclidianDistance(
			CSparseFeatures<float64_t>* l, CSparseFeatures<float64_t>* r);
		virtual ~CSparseEuclidianDistance();

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
		 * @return distance type SPARSEEUCLIDIAN
		 */
		virtual EDistanceType get_distance_type() { return D_SPARSEEUCLIDIAN; }

		/** get supported feature type
		 *
		 * @return feature type DREAL
		 */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }

		/** get name of the distance
		 *
		 * @return name SparseEuclidian
		 */
		virtual const char* get_name() const { return "SparseEuclidianDistance"; }

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
		/*    compute_kernel*/

	private:
		void init();

	protected:
		/** squared left-hand side */
		float64_t* sq_lhs;
		/** squared right-hand side */
		float64_t* sq_rhs;

};

} // namespace shogun
#endif /* _SPARSEEUCLIDIANDISTANCE_H__ */

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _GEODESICMETRIC_H___
#define _GEODESICMETRIC_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class GeodesicMetric
 *
 * The Geodesic distance (Great circle distance) computes the shortest path
 * between two data points on a sphere (the radius is set to one for the
 * evaluation).
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = arccos\sum_{i=1}^{n} \frac{\bf{x_{i}}\cdot\bf{x'_{i}}}
 *  {\sqrt{x_{i}x_{i} x'_{i}x'_{i}}}
 * \f]
 *
 * @see <a href="http://en.wikipedia.org/wiki/Great_circle_distance">Wikipedia:
 * Geodesic distance</a>
 *
 */
class GeodesicMetric: public DenseDistance<float64_t>
{
	public:
		/** default constructor */
		GeodesicMetric();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		GeodesicMetric(std::shared_ptr<DenseFeatures<float64_t>> l, std::shared_ptr<DenseFeatures<float64_t>> r);
		virtual ~GeodesicMetric();

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** cleanup distance */
		virtual void cleanup();

		/** get distance type we are
		 *
		 * @return distance type GEODESIC
		 */
		virtual EDistanceType get_distance_type() { return D_GEODESIC; }

		/** get name of the distance
		 *
		 * @return name Chebyshew-Metric
		 */
		virtual const char* get_name() const { return "GeodesicMetric"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};

} // namespace shogun
#endif /* _GEODESICMETRIC_H___ */

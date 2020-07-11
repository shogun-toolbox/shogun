/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn,
 *          Viktor Gal
 */

#ifndef _CANBERRAMETRIC_H__
#define _CANBERRAMETRIC_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
	template <class T> class DenseFeatures;

/** @brief class CanberraMetric
 *
 * The Canberra distance sums up the dissimilarity (ratios) between feature
 * dimensions of two data points.
 *
 * \f[\displaystyle
 *   d(\bf{x},\bf{x'}) = \sum_{i=1}^{n}\frac{|\bf{x_{i}-\bf{x'_{i}}}|}
 *    {|\bf{x_{i}}|+|\bf{x'_{i}}|} \quad \bf{x},\bf{x'} \in R^{n}
 * \f]
 *
 *  A summation element has range [0,1]. Note that \f$d(x,0)=d(0,x')=n\f$
 *  and \f$d(0,0)=0\f$.
 */
class CanberraMetric: public DenseDistance<float64_t>
{
	public:
		/** default constructor */
		CanberraMetric();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CanberraMetric(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r);
		~CanberraMetric() override;

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** cleanup distance */
		void cleanup() override;

		/** get distance type we are
		 *
		 * @return distance type CANBERRA
		 */
		EDistanceType get_distance_type() override { return D_CANBERRA; }

		/** get name of the distance
		 *
		 * @return name Canberra-Metric
		 */
		const char* get_name() const override { return "CanberraMetric"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b) override;
};

} // namespace shogun
#endif /* _CANBERRAMETRIC_H__ */

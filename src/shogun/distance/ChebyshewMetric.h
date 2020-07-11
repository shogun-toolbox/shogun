/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _CHEBYSHEWMETRIC_H___
#define _CHEBYSHEWMETRIC_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class ChebyshewMetric
 *
 * The Chebyshev distance (\f$L_{\infty}\f$ norm) returns the maximum of
 * absolute feature dimension differences between two data points.
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = max|\bf{x_{i}}-\bf{x'_{i}}| \quad x,x' \in R^{n}
 * \f]
 *
 * @see <a href="http://en.wikipedia.org/wiki/Chebyshev_distance">Wikipedia: Chebyshev distance</a>
 */
class ChebyshewMetric: public DenseDistance<float64_t>
{
	public:
		/** default constructor */
		ChebyshewMetric();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		ChebyshewMetric(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r);
		~ChebyshewMetric() override;

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
		 * @return distance type CHEBYSHEW
		 */
		EDistanceType get_distance_type() override { return D_CHEBYSHEW; }

		/** get name of the distance
		 *
		 * @return name Chebyshew-Metric
		 */
		const char* get_name() const override { return "ChebyshewMetric"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b) override;
};

} // namespace shogun
#endif  /* _CHEBYSHEWMETRIC_H___ */

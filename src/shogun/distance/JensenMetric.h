/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _JENSENMETRIC_H___
#define _JENSENMETRIC_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class JensenMetric
 *
 * The Jensen-Shannon distance/divergence measures the similarity between
 * two data points which is based on the Kullback-Leibler divergence.
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = \sum_{i=0}^{n} x_{i} log\frac{x_{i}}{0.5(x_{i}
 *  +x'_{i})} + x'_{i} log\frac{x'_{i}}{0.5(x_{i}+x'_{i})}
 * \f]
 *
 * @see <a href="http://en.wikipedia.org/wiki/Jensen-Shannon_divergence">
 * Wikipedia: Jensen-Shannon divergence</a>
 * @see <a href="http://en.wikipedia.org/wiki/Kullback-Leibler_divergence">
 * Wikipedia: Kullback-Leibler divergence</a>
 */
class JensenMetric: public DenseDistance<float64_t>
{
	public:
		/** default constructor */
		JensenMetric();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		JensenMetric(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r);
		~JensenMetric() override;

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
		 * @return distance type JENSEN
		 */
		EDistanceType get_distance_type() override { return D_JENSEN; }

		/** get name of the distance
		 *
		 * @return name Jensen-Metric
		 */
		const char* get_name() const override { return "JensenMetric"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b) override;
};
} // namespace shogun

#endif /* _JENSENMETRIC_H___ */
